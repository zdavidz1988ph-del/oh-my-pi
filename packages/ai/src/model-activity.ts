import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";

/** 原生调用方明确绑定的身份，不使用供应方路由键猜测归属。 */
export interface ModelActivityOwner {
	readonly agentId: string;
	readonly sessionId: string;
	readonly rootSessionId: string;
}

export interface ModelActivityRecord {
	readonly id: string;
	readonly agentId?: string;
	readonly sessionId?: string;
	readonly rootSessionId?: string;
	readonly startedAt: number;
	readonly observable: boolean;
}

interface ActivityScope {
	owner?: ModelActivityOwner;
	callId?: string;
	observationId?: string;
	suppressTransport?: boolean;
}

interface ResultStream {
	result(): Promise<unknown>;
}

/** 供本机工作器保留同一请求，调用方取消不等于工作器已经停止。 */
export interface ModelActivityLease {
	finish(): void;
	unobservable(): void;
}

interface ActivityObservation extends ModelActivityLease {
	context: ActivityScope;
}

interface ActiveRecord {
	record: ModelActivityRecord;
	holders: number;
}

const scope = new AsyncLocalStorage<ActivityScope>();
const active = new Map<string, ActiveRecord>();
const unavailable = new Map<string, ModelActivityRecord>();
const pendingTransports = new Map<string, ModelActivityRecord>();
const untrackedTransport: ModelActivityLease = { finish() {}, unobservable() {} };
const MAX_ACTIVE_RECORDS = 4096;
const MAX_UNAVAILABLE_OWNERS = 512;
let unavailableOverflow: ModelActivityRecord | undefined;
export const modelActivityInstanceId = randomUUID();
let nextId = 0;

/** 仅用于隔离测试进程内的用例；真实运行不能清掉未确认结束的事实。 */
export function resetModelActivityForTests(): void {
	active.clear();
	pendingTransports.clear();
	unavailable.clear();
	unavailableOverflow = undefined;
}

/** 子会话、工具和侧聊进入自己的作用域，不继承父请求的去重标记。 */
export function withModelActivityOwner<T>(owner: ModelActivityOwner | undefined, action: () => T): T {
	return scope.run({ owner }, action);
}

/** 在跨工作器分发前捕获归属，迟到的调用不能借用新会话的身份。 */
export function captureModelActivityOwner(): ModelActivityOwner | undefined {
	const owner = scope.getStore()?.owner;
	return owner ? { ...owner } : undefined;
}

/** 返回独立的只读投影；正文、凭据、请求头均不进入记录。 */
export function getModelActivity(): ModelActivityRecord[] {
	return [
		...Array.from(active.values(), entry => ({ ...entry.record })),
		...Array.from(unavailable.values(), record => ({ ...record })),
		...Array.from(pendingTransports.values(), record => ({ ...record })),
		...(unavailableOverflow ? [{ ...unavailableOverflow }] : []),
	];
}

/** 断线后只保留有界身份缺口，不把不可观测的旧请求永久留在活动表里。 */
function rememberUnavailable(record: ModelActivityRecord): void {
	if (unavailableOverflow) return;
	const key = JSON.stringify([record.rootSessionId, record.agentId, record.sessionId]);
	if (unavailable.has(key)) return;
	if (!record.rootSessionId || !record.agentId || !record.sessionId || unavailable.size >= MAX_UNAVAILABLE_OWNERS) {
		unavailable.clear();
		unavailableOverflow = { id: "unavailable", startedAt: Date.now(), observable: false };
		return;
	}
	unavailable.set(key, { ...record, id: `unavailable:${record.id}`, observable: false });
}

function begin(retain: true): ActivityObservation;
function begin(retain?: false): ActivityObservation | undefined;
function begin(retain = false): ActivityObservation | undefined {
	const current = scope.getStore();
	if (current?.observationId) pendingTransports.delete(current.observationId);
	let entry = current?.callId ? active.get(current.callId) : undefined;
	if (entry && !retain) return undefined;
	const owner = current?.owner;
	const id = entry ? entry.record.id : String(++nextId);
	if (!entry) {
		entry = {
			record: {
				id,
				agentId: owner?.agentId,
				sessionId: owner?.sessionId,
				rootSessionId: owner?.rootSessionId,
				startedAt: Date.now(),
				observable: true,
			},
			holders: 0,
		};
		if (active.size >= MAX_ACTIVE_RECORDS) {
			rememberUnavailable(entry.record);
			// 已无法逐条观测；结束通知不能清除已经记录的未知事实。
			return { context: { owner }, finish() {}, unobservable() {} };
		}
		active.set(id, entry);
	}
	const held = entry;
	held.holders++;
	let finished = false;
	return {
		context: { owner, callId: id },
		finish() {
			if (finished) return;
			finished = true;
			if (--held.holders === 0 && active.get(id) === held) active.delete(id);
		},
		unobservable() {
			if (finished) return;
			finished = true;
			rememberUnavailable(held.record);
			held.holders--;
			if (active.get(id) === held) active.delete(id);
		},
	};
}

export function retainModelActivity(): ModelActivityLease {
	return scope.getStore()?.suppressTransport ? untrackedTransport : begin(true);
}

/** 排队时捕获完整归属，真正发送时才开始活动。 */
export function captureModelActivityStarter(): () => ModelActivityLease {
	const context = scope.getStore() ?? {};
	return () => scope.run(context, retainModelActivity);
}

/** 附接既有传输时确认观察已接通，不凭空创建第二个模型请求。 */
export function captureModelTransportObservation(): () => void {
	const id = scope.getStore()?.observationId;
	return id === undefined
		? untrackedTransport.finish
		: () => {
				pendingTransports.delete(id);
			};
}

/** 认证和资料查询不是模型调用；异步完成后恢复调用者的原活动作用域。 */
export function withoutModelTransportActivity<T>(action: () => T): T {
	return scope.run({ ...scope.getStore(), suppressTransport: true }, action);
}

/** 供应方尚未进入可观测传输时只报告缺口，不伪造模型请求编号。 */
export function observeModelTransportStream<T extends ResultStream>(action: () => T): T {
	const current = scope.getStore();
	const owner = current?.owner;
	const id = `pending:${++nextId}`;
	const record: ModelActivityRecord = {
		id,
		agentId: owner?.agentId,
		sessionId: owner?.sessionId,
		rootSessionId: owner?.rootSessionId,
		startedAt: Date.now(),
		observable: false,
	};
	if (pendingTransports.size < MAX_ACTIVE_RECORDS) pendingTransports.set(id, record);
	else rememberUnavailable(record);
	const clear = () => pendingTransports.delete(id);
	try {
		const stream = scope.run({ ...current, observationId: id }, action);
		try {
			void stream.result().then(clear, clear);
		} catch {
			clear();
			rememberUnavailable(record);
		}
		return stream;
	} catch (error) {
		clear();
		throw error;
	}
}

/**
 * 只旁听同一个结果 Promise，不消费事件迭代器，不替换原流。
 * 收到取消信号不等于传输已结束；只有结果完成或创建流失败才结束记录。
 */
export function observeModelStream<T extends ResultStream>(action: () => T): T;
export function observeModelStream<T extends ResultStream>(action: () => T | Promise<T>): T | Promise<T>;
export function observeModelStream<T extends ResultStream>(action: () => T | Promise<T>): T | Promise<T> {
	const activity = begin();
	if (!activity) return action();
	const attach = (stream: T): T => {
		try {
			void stream.result().then(activity.finish, activity.finish);
		} catch {
			// 观测失败不能改变调用结果；未知状态不得冒充空闲。
			activity.unobservable();
		}
		return stream;
	};
	try {
		const result = scope.run(activity.context, action);
		if (result instanceof Promise) {
			return result.then(attach, error => {
				activity.finish();
				throw error;
			});
		}
		return attach(result);
	} catch (error) {
		activity.finish();
		throw error;
	}
}

/** 远端压缩没有消息流，按实际远端操作的 Promise 记录，不包裹本地压缩。 */
export function observeModelOperation<T>(action: () => Promise<T>): Promise<T> {
	const activity = begin();
	if (!activity) return action();
	try {
		return scope.run(activity.context, action).then(
			value => {
				activity.finish();
				return value;
			},
			error => {
				activity.finish();
				throw error;
			},
		);
	} catch (error) {
		activity.finish();
		throw error;
	}
}
