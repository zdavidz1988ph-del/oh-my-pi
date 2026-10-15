import { getModelActivity, modelActivityInstanceId } from "@oh-my-pi/pi-ai";
import { AgentRegistry, type AgentRef } from "../../registry/agent-registry";
import { ensurePersistedRoster, isPersistedRosterReady } from "../../registry/persisted-agents";
import type { AgentSession } from "../../session/agent-session";
import { isRpcSessionSettled, type RpcScheduledTurnProbe } from "./rpc-session-settle";
import type { RpcAgentActivity, RpcRuntimeActivity } from "./rpc-types";

interface PendingQuestions {
	keys(): IterableIterator<string>;
	getOwnerSessionId(id: string): string | undefined;
}

/** 同步读取同一时刻的事实；不依赖显示订阅是否开启，不向代理增加工具。 */
export class RpcRuntimeActivityReader {
	#sequence = 0;
	readonly #rosterLoading = new Map<string, Promise<unknown>>();

	constructor(
		readonly session: AgentSession,
		readonly pending: PendingQuestions,
		readonly registry: AgentRegistry = AgentRegistry.global(),
	) {}

	snapshot(scheduledTurn?: RpcScheduledTurnProbe): RpcRuntimeActivity {
		const sessionId = this.session.sessionId;
		const rootId = this.session.getAgentId();
		const root = rootId ? this.registry.get(rootId) : undefined;
		const gaps = new Set<string>();
		if (this.session.isSessionTransitioning) gaps.add("session_transition");
		const sessionFile = this.session.sessionManager.getSessionFile();
		if (sessionFile && !isPersistedRosterReady(this.registry, sessionFile)) {
			gaps.add("roster_loading");
			if (!this.session.isSessionTransitioning && !this.#rosterLoading.has(sessionFile)) {
				const loading = ensurePersistedRoster(this.registry, sessionFile)
					.catch(() => undefined)
					.finally(() => {
						if (this.#rosterLoading.get(sessionFile) === loading) this.#rosterLoading.delete(sessionFile);
					});
				this.#rosterLoading.set(sessionFile, loading);
			}
		}
		const owned = new Map<string, AgentRef>();
		if (!root || root.session !== this.session || root.kind !== "main") {
			gaps.add("root_identity_unavailable");
		} else {
			owned.set(root.id, root);
			for (const ref of this.registry.list()) {
				if (ref === root || ref.kind === "advisor") continue;
				if (!ref.rootSessionId) gaps.add("unscoped_identity");
				else if (ref.kind === "sub" && ref.rootSessionId === sessionId) owned.set(ref.id, ref);
			}
		}

		const modelIds = new Map<string, string[]>();
		const incompleteModels = new Set<string>();
		let unscopedModels = false;
		for (const call of getModelActivity()) {
			if (!call.rootSessionId || !call.agentId || !call.sessionId) {
				gaps.add("unscoped_model");
				unscopedModels = true;
				continue;
			}
			if (call.rootSessionId !== sessionId) continue;
			const ref = owned.get(call.agentId);
			const actorSessionId = ref?.session?.sessionId ?? ref?.sessionId;
			if (!ref || actorSessionId !== call.sessionId) {
				gaps.add("model_identity_unavailable");
				continue;
			}
			if (!call.observable) {
				incompleteModels.add(ref.id);
				continue;
			}
			let ids = modelIds.get(ref.id);
			if (!ids) modelIds.set(ref.id, (ids = []));
			if (ids.length < 1024) ids.push(call.id);
			else incompleteModels.add(ref.id);
		}

		const ownedSessions = new Set<string>();
		for (const ref of owned.values()) {
			const actorSessionId = ref.session?.sessionId ?? ref.sessionId;
			if (actorSessionId) ownedSessions.add(actorSessionId);
		}
		const questionIds = new Map<string, string[]>();
		for (const id of this.pending.keys()) {
			const owner = this.pending.getOwnerSessionId(id);
			if (!owner || !ownedSessions.has(owner)) {
				gaps.add("question_identity_unavailable");
				continue;
			}
			let ids = questionIds.get(owner);
			if (!ids) questionIds.set(owner, (ids = []));
			ids.push(id);
		}

		const agents: RpcAgentActivity[] = [];
		for (const ref of owned.values()) {
			const live = ref.session;
			const actorSessionId = live?.sessionId ?? ref.sessionId ?? null;
			// 尚在构造的会话不是空闲；停放和取消则由原生名册保证已脱离运行会话。
			let complete = actorSessionId !== null && !unscopedModels && !incompleteModels.has(ref.id);
			if (live) {
				complete &&= live.modelActivityTracked;
				// 现有子代理没有独立 UI。未来有其他 UI 通道时，不假装此表覆盖它。
				if (ref !== root && live.extensionRunner?.hasUI() === true) complete = false;
			} else if (ref.status !== "parked" && ref.status !== "aborted") {
				complete = false;
			}
			if (!complete) gaps.add("agent_activity_incomplete");
			agents.push({
				id: ref.id,
				sessionId: actorSessionId,
				parentId: ref === root ? null : (ref.parentId ?? null),
				name: ref.displayName,
				kind: ref === root ? "main" : "sub",
				status: ref.status,
				modelRequestIds: modelIds.get(ref.id) ?? [],
				toolCallIds: live ? [...live.state.pendingToolCalls] : [],
				pendingQuestionIds: actorSessionId ? (questionIds.get(actorSessionId) ?? []) : [],
				waitingForResult: live
					? live.isBusyForSnapshot ||
						live.isSessionTransitioning ||
						live.state.pendingToolCalls.size > 0 ||
						!isRpcSessionSettled(live, ref === root ? scheduledTurn : undefined)
					: ref.status === "running",
				complete,
			});
		}
		return {
			version: 1,
			instanceId: modelActivityInstanceId,
			sessionId,
			sequence: ++this.#sequence,
			complete: gaps.size === 0,
			gaps: [...gaps],
			agents,
		};
	}
}
