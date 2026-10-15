import { afterEach, beforeEach, expect, test } from "bun:test";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async/job-manager";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentLifecycleManager } from "@oh-my-pi/pi-coding-agent/registry/agent-lifecycle";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { VIBE_LIFECYCLE_CUSTOM_TYPE } from "@oh-my-pi/pi-coding-agent/vibe/lifecycle";
import { VibeSessionRegistry } from "@oh-my-pi/pi-coding-agent/vibe/runtime";
import { TempDir } from "@oh-my-pi/pi-utils";

let directory: TempDir;
let manager: SessionManager;
let jobs: AsyncJobManager;
let parent: ToolSession;
let childSessionId: string;
let childSessionFile: string;
let lifecycle: { version: 1; id: string; ownerId: string; parentSessionId: string };

beforeEach(async () => {
	AgentRegistry.resetGlobalForTests();
	AgentLifecycleManager.resetGlobalForTests();
	VibeSessionRegistry.resetGlobalForTests();
	directory = TempDir.createSync("omp-vibe-activity-identity-");
	manager = SessionManager.create(directory.path(), directory.path());
	jobs = new AsyncJobManager({ onJobComplete: () => {} });
	parent = {
		cwd: directory.path(), settings: Settings.isolated(), sessionManager: manager, asyncJobManager: jobs,
		getAgentId: () => "Main", getSessionId: () => manager.getSessionId(),
		getRootSessionId: () => manager.getSessionId(), getSessionFile: () => manager.getSessionFile(),
	} as unknown as ToolSession;
	const child = SessionManager.create(directory.path(), path.join(directory.path(), "source"));
	child.appendSessionInit({ systemPrompt: [], task: "本机恢复验证", tools: [], agent: "sonic" });
	await child.ensureOnDisk();
	childSessionId = child.getSessionId();
	const source = child.getSessionFile()!;
	await child.close();
	childSessionFile = path.resolve(manager.getSessionFile()!.slice(0, -6), "Restored.jsonl");
	await fs.mkdir(path.dirname(childSessionFile), { recursive: true });
	await fs.copyFile(source, childSessionFile);
	lifecycle = { version: 1, id: "Restored", ownerId: "Main", parentSessionId: manager.getSessionId() };
	manager.appendCustomEntry(VIBE_LIFECYCLE_CUSTOM_TYPE, {
		...lifecycle, action: "spawn", cli: "fast", agent: "sonic", childSessionFile: "Restored.jsonl", createdAt: 1,
	});
	manager.appendCustomEntry(VIBE_LIFECYCLE_CUSTOM_TYPE, { ...lifecycle, action: "turn-started", turn: 1 });
	manager.appendCustomEntry(VIBE_LIFECYCLE_CUSTOM_TYPE, { ...lifecycle, action: "turn-settled", turn: 1 });
});

afterEach(async () => {
	await jobs.dispose({ timeoutMs: 100 });
	await manager.close();
	AgentLifecycleManager.resetGlobalForTests();
	AgentRegistry.resetGlobalForTests();
	VibeSessionRegistry.resetGlobalForTests();
	await directory.remove();
});

const expectedIdentity = () => ({ rootSessionId: manager.getSessionId(), sessionId: childSessionId });

test("从真实日志恢复的持久成员保留原会话编号，取消后仍可对应同一人", async () => {
	const registry = VibeSessionRegistry.global();
	expect(await registry.rehydrate(parent)).toBe(1);
	expect(AgentRegistry.global().get("Restored")).toMatchObject({ ...expectedIdentity(), status: "parked" });
	await registry.kill(parent, "Restored");
	expect(AgentRegistry.global().get("Restored")).toMatchObject({ ...expectedIdentity(), status: "aborted" });
});

test("重启后恢复已取消成员时同样读取真实身份，不凭人物名制造会话编号", async () => {
	manager.appendCustomEntry(VIBE_LIFECYCLE_CUSTOM_TYPE, { ...lifecycle, action: "tombstone", reason: "explicit-kill" });
	expect(await VibeSessionRegistry.global().rehydrate(parent)).toBe(0);
	expect(AgentRegistry.global().get("Restored")).toMatchObject({ ...expectedIdentity(), status: "aborted" });
});

test("同路径的旧停放记录可补齐身份，但不能覆盖另一个根会话的成员", async () => {
	const registry = AgentRegistry.global();
	const parked = registry.register({ id: "Restored", displayName: "Restored", kind: "sub", parentId: "Main",
		session: null, sessionFile: childSessionFile, status: "parked" });
	await VibeSessionRegistry.global().rehydrate(parent);
	expect(registry.get("Restored")).toBe(parked);
	expect(parked).toMatchObject(expectedIdentity());
	VibeSessionRegistry.resetGlobalForTests();
	parked.rootSessionId = "another-root";
	parked.sessionId = "another-session";
	manager.appendCustomEntry(VIBE_LIFECYCLE_CUSTOM_TYPE, { ...lifecycle, action: "tombstone", reason: "explicit-kill" });
	await VibeSessionRegistry.global().rehydrate(parent);
	expect(registry.get("Restored")).toBe(parked);
	expect(parked).toMatchObject({ rootSessionId: "another-root", sessionId: "another-session", status: "parked" });
});
