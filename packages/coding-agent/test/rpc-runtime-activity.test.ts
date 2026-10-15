import { afterEach, beforeEach, expect, test } from "bun:test";
import * as path from "node:path";
import { Agent, type StreamFn } from "@oh-my-pi/pi-agent-core";
import { observeModelStream, retainModelActivity, withModelActivityOwner } from "@oh-my-pi/pi-ai";
import { resetModelActivityForTests } from "@oh-my-pi/pi-ai/model-activity";
import { EventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { RpcRuntimeActivityReader } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-runtime-activity";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { ensurePersistedRoster } from "@oh-my-pi/pi-coding-agent/registry/persisted-agents";
import { CURRENT_SESSION_VERSION } from "@oh-my-pi/pi-coding-agent/session/session-entries";
import { createAgentSession } from "@oh-my-pi/pi-coding-agent/sdk";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createInMemoryAuthStorage } from "./helpers/agent-session-setup";

const resources: Array<() => Promise<void> | void> = [];
// 其他协议测试会故意制造进程级失联；本夹具从独立的空活动表开始。
beforeEach(() => resetModelActivityForTests());
afterEach(async () => {
	for (const dispose of resources.reverse()) await dispose();
	resources.length = 0;
	resetModelActivityForTests();
});

function fixture(mainManager?: SessionManager) {
	const temp = TempDir.createSync("@rpc-runtime-activity-");
	resources.push(() => temp.removeSync());
	const auth = createInMemoryAuthStorage();
	resources.push(() => auth.close());
	const models = new ModelRegistry(auth, path.join(temp.path(), "models.yml"));
	const registry = new AgentRegistry();
	const model = getBundledModel("anthropic", "claude-sonnet-4-5");
	if (!model) throw new Error("缺少测试模型定义");
	const questions = new Map<string, string>();
	const create = (id: string, rootSessionId?: string) => {
		const manager = id === "Main" && mainManager ? mainManager : SessionManager.inMemory(temp.path());
		const root = rootSessionId ?? manager.getSessionId();
		const scope = <T>(action: () => T) =>
			withModelActivityOwner({ agentId: id, sessionId: manager.getSessionId(), rootSessionId: root }, action);
		const streamFn: StreamFn = () => {
			throw new Error("本测试不允许调用外部模型");
		};
		const agent = new Agent({ initialState: { model, tools: [] }, streamFn, runInModelActivityScope: scope });
		const session = new AgentSession({
			agent,
			sessionManager: manager,
			settings: Settings.isolated({ "compaction.enabled": false }),
			modelRegistry: models,
			agentId: id,
			agentKind: rootSessionId ? "sub" : "main",
			rootSessionId: root,
			runInModelActivityScope: scope,
			modelActivityStreamFn: streamFn,
		});
		resources.push(() => session.dispose());
		registry.register({
			id,
			displayName: id,
			kind: rootSessionId ? "sub" : "main",
			parentId: rootSessionId ? "Main" : undefined,
			rootSessionId: root,
			sessionId: session.sessionId,
			session,
		});
		return session;
	};
	const main = create("Main");
	const reader = new RpcRuntimeActivityReader(
		main,
		{ keys: () => questions.keys(), getOwnerSessionId: id => questions.get(id) },
		registry,
	);
	return { main, create, registry, questions, reader };
}

test("两个真实子身份分开记录；模型结束后工具和待答仍然保留", async () => {
	const { main, create, reader, questions } = fixture();
	const a = create("child-a", main.sessionId);
	const b = create("child-b", main.sessionId);
	const first = new EventStream<number, number>(
		() => true,
		value => value,
	);
	const second = new EventStream<number, number>(
		() => true,
		value => value,
	);
	a.runInModelActivityScope(() => observeModelStream(() => first));
	b.runInModelActivityScope(() => observeModelStream(() => second));
	try {
		let snapshot = reader.snapshot();
		expect(snapshot.complete).toBe(true);
		expect(snapshot.agents.filter(agent => agent.modelRequestIds.length > 0).map(agent => agent.id)).toEqual([
			"child-a",
			"child-b",
		]);
		a.agent.state.pendingToolCalls.add("test-command");
		questions.set("question-a", a.sessionId);
		first.end(1);
		await first.result();
		snapshot = reader.snapshot();
		expect(snapshot.agents.find(agent => agent.id === "child-a")).toMatchObject({
			modelRequestIds: [],
			toolCallIds: ["test-command"],
			pendingQuestionIds: ["question-a"],
		});
		expect(snapshot.agents.find(agent => agent.id === "child-b")?.modelRequestIds).toHaveLength(1);
		expect(snapshot.agents.find(agent => agent.id === "Main")?.pendingQuestionIds).toEqual([]);
		a.agent.state.pendingToolCalls.clear();
		questions.delete("question-a");
	} finally {
		first.end(1);
		second.end(2);
		await Promise.all([first.result(), second.result()]);
		a.agent.state.pendingToolCalls.clear();
	}
});

test("停放身份仍保留，但其他根会话和未取得归属的小人不会被混入", () => {
	const { main, create, registry, reader } = fixture();
	const child = create("child", main.sessionId);
	const ref = registry.get("child")!;
	ref.session = null;
	registry.setStatus("child", "parked", ref);
	create("other-child", "other-root");
	let snapshot = reader.snapshot();
	expect(snapshot.complete).toBe(true);
	expect(snapshot.agents.map(agent => agent.id)).toEqual(["Main", "child"]);
	expect(snapshot.agents[1]).toMatchObject({ sessionId: child.sessionId, status: "parked", waitingForResult: false });
	registry.register({ id: "unknown", displayName: "未确认", kind: "sub", session: null });
	snapshot = reader.snapshot();
	expect(snapshot.complete).toBe(false);
	expect(snapshot.gaps).toContain("unscoped_identity");
	expect(snapshot.agents.map(agent => agent.id)).toEqual(["Main", "child"]);
});

test("同名代理换了会话后，旧模型不能落到新代理身上", async () => {
	const { main, create, reader } = fixture();
	const old = create("child", main.sessionId);
	const running = new EventStream<number, number>(
		() => true,
		value => value,
	);
	old.runInModelActivityScope(() => observeModelStream(() => running));
	try {
		const replacement = create("child", main.sessionId);
		const snapshot = reader.snapshot();
		expect(snapshot.complete).toBe(false);
		expect(snapshot.gaps).toContain("model_identity_unavailable");
		expect(snapshot.agents.find(agent => agent.id === "child")).toMatchObject({
			sessionId: replacement.sessionId,
			modelRequestIds: [],
		});
	} finally {
		running.end(1);
		await running.result();
	}
	expect(reader.snapshot().complete).toBe(true);
});

test("未跟踪的自定义流与尚未接好的会话必须显示不完整，而非空闲", () => {
	const { main, registry, reader } = fixture();
	main.agent.streamFn = () => {
		throw new Error("未接入活动的自定义流");
	};
	registry.register({
		id: "constructing",
		displayName: "正在接入",
		kind: "sub",
		rootSessionId: main.sessionId,
		session: null,
	});
	const snapshot = reader.snapshot();
	expect(snapshot.complete).toBe(false);
	expect(snapshot.agents.find(agent => agent.id === "Main")?.complete).toBe(false);
	expect(snapshot.agents.find(agent => agent.id === "constructing")).toMatchObject({
		complete: false,
		waitingForResult: true,
		sessionId: null,
	});
});

test("真实 SDK 创建的空闲主会话不会被包装函数误报为未知", async () => {
	const temp = TempDir.createSync("@rpc-sdk-activity-");
	resources.push(() => temp.removeSync());
	const auth = createInMemoryAuthStorage();
	resources.push(() => auth.close());
	const registry = new AgentRegistry();
	const models = new ModelRegistry(auth, path.join(temp.path(), "models.yml"));
	const { session } = await createAgentSession({
		cwd: temp.path(),
		sessionManager: SessionManager.inMemory(temp.path()),
		authStorage: auth,
		modelRegistry: models,
		agentRegistry: registry,
		model: getBundledModel("anthropic", "claude-sonnet-4-5")!,
		settings: Settings.isolated({ "compaction.enabled": false, "skills.enabled": false }),
		toolNames: [],
		enableMCP: false,
		enableLsp: false,
		disableExtensionDiscovery: true,
	});
	resources.push(() => session.dispose());
	const reader = new RpcRuntimeActivityReader(
		session,
		{ keys: () => new Map<string, string>().keys(), getOwnerSessionId: () => undefined },
		registry,
	);
	const snapshot = reader.snapshot();
	expect(snapshot.agents).toMatchObject([
		{ id: "Main", complete: true, modelRequestIds: [], waitingForResult: false },
	]);
	expect(snapshot.gaps).not.toContain("agent_activity_incomplete");
	session.agent.streamFn = () => {
		throw new Error("未接入记录的替换入口");
	};
	expect(reader.snapshot().agents[0]?.complete).toBe(false);
});

test("反复失联仅报告缺口，不把旧请求塞满快照或冒充模型仍在运行", () => {
	const { main, reader } = fixture();
	for (let i = 0; i < 1500; i++) main.runInModelActivityScope(() => retainModelActivity().unobservable());
	const snapshot = reader.snapshot();
	expect(snapshot.complete).toBe(false);
	expect(snapshot.agents[0]).toMatchObject({ id: "Main", complete: false, modelRequestIds: [] });
	expect(snapshot.gaps).toContain("agent_activity_incomplete");
});

test("真实并发请求超过接收上限时保留有界编号并明确不完整", () => {
	const { main, reader } = fixture();
	const leases = Array.from({ length: 1100 }, () => main.runInModelActivityScope(retainModelActivity));
	try {
		const snapshot = reader.snapshot();
		expect(snapshot.complete).toBe(false);
		expect(snapshot.agents[0]?.modelRequestIds).toHaveLength(1024);
		expect(snapshot.agents[0]?.complete).toBe(false);
	} finally {
		for (const lease of leases) lease.finish();
	}
	expect(reader.snapshot().agents[0]).toMatchObject({ complete: true, modelRequestIds: [] });
});

test("RPC 对未恢复的真实 Vibe 日志报告缺口，准确登记后才完整", async () => {
	const temp = TempDir.createSync("@rpc-vibe-roster-");
	resources.push(() => temp.removeSync());
	const rootFile = path.join(temp.path(), "main.jsonl");
	const childFile = path.join(temp.path(), "main", "Worker.jsonl");
	const header = (id: string) =>
		JSON.stringify({
			type: "session",
			version: CURRENT_SESSION_VERSION,
			id,
			timestamp: "2026-08-25T10:00:00.000Z",
			cwd: temp.path(),
		});
	await Bun.write(
		rootFile,
		`${header("vibe-rpc-root")}\n${JSON.stringify({
			type: "custom",
			id: "spawn",
			parentId: null,
			timestamp: "2026-08-25T10:00:01.000Z",
			customType: "vibe-session-lifecycle",
			data: {
				version: 1,
				action: "spawn",
				id: "Worker",
				ownerId: "Main",
				parentSessionId: "vibe-rpc-root",
				cli: "fast",
				agent: "task",
				childSessionFile: "Worker.jsonl",
				createdAt: 1,
			},
		})}\n`,
	);
	await Bun.write(
		childFile,
		`${header("vibe-rpc-child")}\n${JSON.stringify({
			type: "session_init",
			id: "init",
			parentId: null,
			timestamp: "2026-08-25T10:00:01.000Z",
			systemPrompt: "检查",
			task: "检查",
			tools: ["read"],
		})}\n`,
	);
	const { reader, registry } = fixture(await SessionManager.open(rootFile));
	await ensurePersistedRoster(registry, rootFile);
	expect(reader.snapshot()).toMatchObject({ complete: false, gaps: ["roster_loading"] });
	expect(registry.get("Worker")).toBeUndefined();
	registry.register({
		id: "Worker",
		displayName: "Worker",
		kind: "sub",
		parentId: "Main",
		rootSessionId: "vibe-rpc-root",
		sessionId: "vibe-rpc-child",
		session: null,
		sessionFile: childFile,
		status: "parked",
	});
	await ensurePersistedRoster(registry, rootFile);
	const snapshot = reader.snapshot();
	expect(snapshot.complete).toBe(true);
	expect(snapshot.gaps).toEqual([]);
	expect(snapshot.agents.find(actor => actor.id === "Worker")).toMatchObject({
		sessionId: "vibe-rpc-child",
		status: "parked",
		complete: true,
		modelRequestIds: [],
		toolCallIds: [],
		pendingQuestionIds: [],
		waitingForResult: false,
	});
	registry.get("Worker")!.sessionId = "wrong";
	expect(reader.snapshot().complete).toBe(false);
	await ensurePersistedRoster(registry, rootFile);
});
