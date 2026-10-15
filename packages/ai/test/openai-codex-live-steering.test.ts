import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as AIError from "@oh-my-pi/pi-ai/error";
import { getModelActivity, resetModelActivityForTests, withModelActivityOwner } from "@oh-my-pi/pi-ai/model-activity";
import { planSteeredRequest } from "@oh-my-pi/pi-ai/providers/openai-codex/live-steering";
import { streamOpenAICodexResponses } from "@oh-my-pi/pi-ai/providers/openai-codex-responses";
import type {
	AssistantMessage,
	Context,
	FetchImpl,
	LiveSteerClaim,
	LiveSteering,
	Model,
	ProviderSessionState,
	ToolResultMessage,
	UserMessage,
} from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import * as piUtils from "@oh-my-pi/pi-utils";

const { getAgentDir, setAgentDir, TempDir } = piUtils;
const originalAgentDir = getAgentDir();
const originalWebSocket = global.WebSocket;
const USAGE = { input_tokens: 5, output_tokens: 3, total_tokens: 8, input_tokens_details: { cached_tokens: 0 } };

beforeEach(() => {
	// 上限测试会留下进程级未知；每个协议场景独立开始，场景内不得清表掩盖残留。
	resetModelActivityForTests();
	setAgentDir(TempDir.createSync("@pi-codex-steer-").path());
	vi.spyOn(piUtils, "getInstallId").mockReturnValue("00000000-0000-4000-8000-000000000001");
});

afterEach(() => {
	global.WebSocket = originalWebSocket;
	setAgentDir(originalAgentDir);
	vi.restoreAllMocks();
});

type Frame = Record<string, unknown>;

/** Scripted Responses WebSocket: `onFrame` answers each client frame via `emit`. */
class ScriptedWebSocket {
	static readonly CONNECTING = 0;
	static readonly OPEN = 1;
	static readonly CLOSING = 2;
	static readonly CLOSED = 3;
	static instances: ScriptedWebSocket[] = [];
	static onFrame: (frame: Frame, socket: ScriptedWebSocket) => void = () => {};
	static sent: Frame[] = [];

	readyState = ScriptedWebSocket.CONNECTING;
	binaryType = "nodebuffer";
	onopen: ((event: Event) => void) | null = null;
	onmessage: ((event: MessageEvent) => void) | null = null;
	onerror: ((event: Event) => void) | null = null;
	onclose: ((event: Event) => void) | null = null;

	constructor() {
		ScriptedWebSocket.instances.push(this);
		queueMicrotask(() => {
			this.readyState = ScriptedWebSocket.OPEN;
			this.onopen?.(new Event("open"));
		});
	}

	send(data: string): void {
		const frame = JSON.parse(data) as Frame;
		ScriptedWebSocket.sent.push(frame);
		ScriptedWebSocket.onFrame(frame, this);
	}

	close(): void {
		this.readyState = ScriptedWebSocket.CLOSED;
	}

	/** Deliver server frames asynchronously, in order. */
	emit(...frames: Frame[]): void {
		queueMicrotask(() => {
			for (const frame of frames) this.onmessage?.({ data: JSON.stringify(frame) } as MessageEvent);
		});
	}
}

function installSocket(onFrame: (frame: Frame, socket: ScriptedWebSocket) => void): void {
	ScriptedWebSocket.instances = [];
	ScriptedWebSocket.sent = [];
	ScriptedWebSocket.onFrame = onFrame;
	global.WebSocket = ScriptedWebSocket as unknown as typeof WebSocket;
}

function createGpt6Model(): Model<"openai-codex-responses"> {
	return buildModel({
		id: "gpt-6-sol",
		name: "GPT-6 Sol",
		api: "openai-codex-responses",
		provider: "openai-codex",
		baseUrl: "https://chatgpt.com/backend-api",
		reasoning: true,
		preferWebsockets: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 272000,
		maxTokens: 128000,
	});
}

function createToken(): string {
	const payload = Buffer.from(
		JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "acc_test" } }),
		"utf8",
	).toBase64();
	return `aaa.${payload}.bbb`;
}

function messageFrames(itemId: string, text: string): Frame[] {
	const item = { type: "message", id: itemId, role: "assistant", content: [{ type: "output_text", text }] };
	return [
		{ type: "response.output_item.added", item: { ...item, status: "in_progress", content: [] } },
		{ type: "response.output_text.delta", item_id: itemId, delta: text },
		{ type: "response.output_item.done", item: { ...item, status: "completed" } },
	];
}

/** A source holding one steering message; records how the provider settled it. */
function oneShotSteering(text: string): { source: LiveSteering; settled: () => "accepted" | "rejected" | undefined } {
	let claimed = false;
	let outcome: "accepted" | "rejected" | undefined;
	const idle = (signal: AbortSignal) => {
		const { promise, resolve } = Promise.withResolvers<void>();
		if (signal.aborted) resolve();
		else signal.addEventListener("abort", () => resolve(), { once: true });
		return promise;
	};
	const source: LiveSteering = {
		wait: async signal => (claimed ? idle(signal) : undefined),
		claim: async (): Promise<LiveSteerClaim | undefined> => {
			if (claimed) return undefined;
			claimed = true;
			return {
				messages: [{ role: "user", content: text, timestamp: Date.now() }],
				accept: () => {
					outcome = "accepted";
				},
				reject: () => {
					outcome = "rejected";
				},
			};
		},
	};
	return { source, settled: () => outcome };
}

function options(providerSessionState: Map<string, ProviderSessionState>, liveSteering?: LiveSteering) {
	return {
		apiKey: createToken(),
		sessionId: "ws-steer-session",
		providerSessionState,
		liveSteering,
		fetch: vi.fn(async () => {
			throw new Error("SSE fallback should not run");
		}) as FetchImpl,
	};
}

const SYSTEM = ["You are a helpful assistant."];
const creates = () => ScriptedWebSocket.sent.filter(frame => frame.type === "response.create");
const steers = () => ScriptedWebSocket.sent.filter(frame => frame.type === "response.steer");

describe("codex live steering", () => {
	it("steers the streaming response and reads the server's automatic continuation without sending a request", async () => {
		installSocket((frame, socket) => {
			if (frame.type === "response.create") {
				// Stream part of an answer, then wait for the steer.
				socket.emit({ type: "response.created", response: { id: "resp_1" } }, ...messageFrames("msg_1", "Plan"));
				return;
			}
			socket.emit(
				{ type: "response.steer.accepted", steer: { id: "steer_1", previous_response_id: "resp_1" } },
				{
					type: "response.incomplete",
					response: {
						id: "resp_1",
						status: "incomplete",
						incomplete_details: { reason: "steered" },
						usage: USAGE,
					},
				},
				// The successor the server creates on its own from the queued input.
				{ type: "response.created", response: { id: "resp_2" } },
				...messageFrames("msg_2", "Tabs it is"),
				{ type: "response.completed", response: { id: "resp_2", status: "completed", usage: USAGE } },
			);
		});
		const model = createGpt6Model();
		const state = new Map<string, ProviderSessionState>();
		const steering = oneShotSteering("use tabs");
		const user: UserMessage = { role: "user", content: "Draft a plan", timestamp: Date.now() };

		const first = await streamOpenAICodexResponses(
			model,
			{ systemPrompt: SYSTEM, messages: [user] },
			options(state, steering.source),
		).result();

		expect(steering.settled()).toBe("accepted");
		expect(first.stopReason).toBe("stop");
		expect(steers()).toEqual([
			{
				type: "response.steer",
				previous_response_id: "resp_1",
				input: [{ role: "user", content: [{ type: "input_text", text: "use tabs" }] }],
			},
		]);

		const steerMessage: UserMessage = { role: "user", content: "use tabs", timestamp: Date.now() };
		const second = await streamOpenAICodexResponses(
			model,
			{ systemPrompt: SYSTEM, messages: [user, first, steerMessage] },
			options(state),
		).result();

		expect(second.responseId).toBe("resp_2");
		expect(second.content).toEqual([expect.objectContaining({ type: "text", text: "Tabs it is" })]);
		expect(creates()).toHaveLength(1);
		expect(ScriptedWebSocket.instances).toHaveLength(1);
	});

	it.each([
		["function_call", false],
		["custom_tool_call", false],
		["function_call", true],
		["custom_tool_call", true],
	] as const)(
		"工具结果接续保留真实活动：%s，发送失败=%s",
		async (callType, failSend) => {
			const activityBefore = getModelActivity();
			const rootSessionId = `steer-tool-${callType}-${failSend}`;
			const owner = failSend ? { agentId: "Main", sessionId: rootSessionId, rootSessionId } : undefined;
			const call =
				callType === "custom_tool_call"
					? { type: callType, id: "ctc_1", call_id: "call_1", name: "eval", input: "print(1)" }
					: { type: callType, id: "fc_1", call_id: "call_1", name: "status", arguments: "{}" };
			installSocket((frame, socket) => {
				if (frame.type === "response.create" && creates().length === 1) {
					socket.emit(
						{ type: "response.created", response: { id: "resp_1" } },
						{
							type: "response.output_item.added",
							output_index: 0,
							item: {
								...call,
								...(callType === "custom_tool_call" ? { input: "" } : { arguments: "" }),
							},
						},
					);
					return;
				}
				if (frame.type === "response.steer") {
					socket.emit(
						{ type: "response.steer.accepted", steer: { id: "steer_1", previous_response_id: "resp_1" } },
						{ type: "response.output_item.done", output_index: 0, item: { ...call, status: "completed" } },
						{ type: "response.completed", response: { id: "resp_1", status: "completed", usage: USAGE } },
						{
							type: "response.steer.pending",
							steer: { id: "steer_1", previous_response_id: "resp_1" },
							reason: "waiting_for_required_input",
							required_input: [{ type: `${callType}_output`, call_id: "call_1", name: call.name }],
						},
					);
					return;
				}
				if (failSend) throw new Error("websocket error: 模拟工具结果尚未送出时连接失败");
				socket.emit(
					{ type: "response.created", response: { id: "resp_2" } },
					...messageFrames("msg_2", "Scoped down"),
					{ type: "response.completed", response: { id: "resp_2", status: "completed", usage: USAGE } },
				);
			});
			const model = { ...createGpt6Model(), applyPatchToolType: "freeform" as const };
			const state = new Map<string, ProviderSessionState>();
			const steering = oneShotSteering("keep it small");
			const user: UserMessage = { role: "user", content: "Plan the project", timestamp: Date.now() };

			const first = await withModelActivityOwner(owner, () =>
				streamOpenAICodexResponses(
					model,
					{ systemPrompt: SYSTEM, messages: [user] },
					options(state, steering.source),
				),
			).result();
			expect(steering.settled()).toBe("accepted");
			expect(first.stopReason).toBe("toolUse");
			expect(getModelActivity().filter(record => !activityBefore.some(old => old.id === record.id))).toEqual([
				expect.objectContaining({ observable: false }),
			]);
			const toolCall = first.content.find(block => block.type === "toolCall");
			if (toolCall?.type !== "toolCall") throw new Error("expected a tool call");
			const toolResult: ToolResultMessage = {
				role: "toolResult",
				toolCallId: toolCall.id,
				toolName: toolCall.name,
				content: [{ type: "text", text: "design done" }],
				isError: false,
				timestamp: Date.now(),
			};
			const steerMessage: UserMessage = { role: "user", content: "keep it small", timestamp: Date.now() };

			const second = await withModelActivityOwner(owner, () =>
				streamOpenAICodexResponses(
					model,
					{ systemPrompt: SYSTEM, messages: [user, first, toolResult, steerMessage] },
					{ ...options(state), codexSseMaxAttempts: 1 },
				),
			).result();
			if (failSend) {
				expect(second.stopReason).toBe("error");
				expect(getModelActivity().filter(record => record.rootSessionId === rootSessionId)).toEqual([
					expect.objectContaining({ ...owner, observable: false }),
				]);
			} else {
				expect(second.stopReason).toBe("stop");
				expect(getModelActivity()).toEqual(activityBefore);
			}

			const continuation = creates()[1];
			expect(continuation?.previous_response_id).toBe("resp_1");
			expect(continuation?.input).toEqual([
				expect.objectContaining({ type: `${callType}_output`, call_id: "call_1", output: "design done" }),
			]);
		},
		30_000,
	);

	it("hands rejected steering back and sends it as ordinary input next time", async () => {
		installSocket((frame, socket) => {
			if (frame.type === "response.create" && creates().length === 1) {
				socket.emit({ type: "response.created", response: { id: "resp_1" } }, ...messageFrames("msg_1", "Hi"));
				return;
			}
			if (frame.type === "response.steer") {
				socket.emit(
					{
						type: "response.steer.failed",
						steer: { previous_response_id: "resp_1", input: frame.input },
						error: { type: "invalid_request_error", code: "response_already_completed", message: "done" },
					},
					{ type: "response.completed", response: { id: "resp_1", status: "completed", usage: USAGE } },
				);
				return;
			}
			socket.emit({ type: "response.created", response: { id: "resp_2" } }, ...messageFrames("msg_2", "Sure"), {
				type: "response.completed",
				response: { id: "resp_2", status: "completed", usage: USAGE },
			});
		});
		const model = createGpt6Model();
		const state = new Map<string, ProviderSessionState>();
		const steering = oneShotSteering("one more thing");
		const user: UserMessage = { role: "user", content: "Hello", timestamp: Date.now() };

		const first: AssistantMessage = await streamOpenAICodexResponses(
			model,
			{ systemPrompt: SYSTEM, messages: [user] },
			options(state, steering.source),
		).result();
		expect(steering.settled()).toBe("rejected");

		const steerMessage: UserMessage = { role: "user", content: "one more thing", timestamp: Date.now() };
		const context: Context = { systemPrompt: SYSTEM, messages: [user, first, steerMessage] };
		await streamOpenAICodexResponses(model, context, options(state)).result();

		expect(creates()[1]?.previous_response_id).toBe("resp_1");
		expect(creates()[1]?.input).toEqual([
			{ role: "user", content: [{ type: "input_text", text: "one more thing" }] },
		]);
	});

	it("replays without steering when the native turn lane rejects it", async () => {
		installSocket((frame, socket) => {
			if (frame.type === "response.create" && creates().length === 1) {
				socket.emit({ type: "response.created", response: { id: "resp_1" } });
				return;
			}
			if (frame.type === "response.steer") {
				// The lane refuses the steer, drops the response, and closes the socket.
				socket.emit({
					type: "error",
					error: {
						type: "invalid_request_error",
						code: "unsupported_native_inflight_message",
						message: "The experimental native turn lane cannot accept stateful WebSocket messages.",
					},
				});
				queueMicrotask(() => {
					socket.readyState = ScriptedWebSocket.CLOSED;
					socket.onclose?.(new CloseEvent("close", { code: 1000 }));
				});
				return;
			}
			const id = `resp_${creates().length}`;
			socket.emit({ type: "response.created", response: { id } }, ...messageFrames(`msg_${id}`, "Hi"), {
				type: "response.completed",
				response: { id, status: "completed", usage: USAGE },
			});
		});
		const model = createGpt6Model();
		const state = new Map<string, ProviderSessionState>();
		const steering = oneShotSteering("one more thing");
		const user: UserMessage = { role: "user", content: "Hello", timestamp: Date.now() };

		const first = await streamOpenAICodexResponses(
			model,
			{ systemPrompt: SYSTEM, messages: [user] },
			options(state, steering.source),
		).result();
		expect(first.stopReason).toBe("stop");
		expect(first.content).toEqual([expect.objectContaining({ type: "text", text: "Hi" })]);
		expect(steering.settled()).toBe("rejected");

		// The session stops steering: a later response is not steered again.
		const steerMessage: UserMessage = { role: "user", content: "one more thing", timestamp: Date.now() };
		const next = oneShotSteering("and another");
		await streamOpenAICodexResponses(
			model,
			{ systemPrompt: SYSTEM, messages: [user, first, steerMessage] },
			options(state, next.source),
		).result();
		expect(next.settled()).toBeUndefined();
		expect(steers()).toHaveLength(1);
		expect(creates()).toHaveLength(3);
	});

	it("leaves a native-lane rejection after streamed reasoning to the agent retry, which no longer steers", async () => {
		installSocket((frame, socket) => {
			if (frame.type === "response.create" && creates().length === 1) {
				// Reasoning streams before the steer goes out.
				socket.emit(
					{ type: "response.created", response: { id: "resp_1" } },
					{ type: "response.output_item.added", item: { type: "reasoning", id: "rs_1", summary: [] } },
					{
						type: "response.reasoning_summary_part.added",
						item_id: "rs_1",
						summary_index: 0,
						part: { type: "summary_text", text: "" },
					},
					{ type: "response.reasoning_summary_text.delta", item_id: "rs_1", summary_index: 0, delta: "Planning" },
				);
				return;
			}
			if (frame.type === "response.steer") {
				socket.emit({
					type: "error",
					error: {
						type: "invalid_request_error",
						code: "unsupported_native_inflight_message",
						message: "The experimental native turn lane cannot accept stateful WebSocket messages.",
					},
				});
				queueMicrotask(() => {
					socket.readyState = ScriptedWebSocket.CLOSED;
					socket.onclose?.(new CloseEvent("close", { code: 1000 }));
				});
				return;
			}
			const id = `resp_${creates().length}`;
			socket.emit({ type: "response.created", response: { id } }, ...messageFrames(`msg_${id}`, "Done"), {
				type: "response.completed",
				response: { id, status: "completed", usage: USAGE },
			});
		});
		const model = createGpt6Model();
		const state = new Map<string, ProviderSessionState>();
		const steering = oneShotSteering("one more thing");
		const user: UserMessage = { role: "user", content: "Hello", timestamp: Date.now() };

		const first = await streamOpenAICodexResponses(
			model,
			{ systemPrompt: SYSTEM, messages: [user] },
			options(state, steering.source),
		).result();
		// Replaying here would re-emit the streamed reasoning, so the provider
		// surfaces a retryable rejection for the agent's same-model retry.
		expect(first.stopReason).toBe("error");
		expect(first.content).toEqual([expect.objectContaining({ type: "thinking", thinking: "Planning" })]);
		expect(AIError.isCodexSteerRejection(first)).toBe(true);
		expect(AIError.retriable(AIError.classifyMessage(first))).toBe(true);
		expect(steering.settled()).toBe("rejected");
		expect(creates()).toHaveLength(1);

		// That retry drops the failed turn and carries the handed-back steer as
		// ordinary input on a fresh chain; the session no longer steers.
		const steerMessage: UserMessage = { role: "user", content: "one more thing", timestamp: Date.now() };
		const next = oneShotSteering("and another");
		const retry = await streamOpenAICodexResponses(
			model,
			{ systemPrompt: SYSTEM, messages: [user, steerMessage] },
			options(state, next.source),
		).result();
		expect(retry.stopReason).toBe("stop");
		expect(next.settled()).toBeUndefined();
		expect(steers()).toHaveLength(1);
		expect(creates()[1]?.previous_response_id).toBeUndefined();
		expect(creates()[1]?.input).toEqual([
			{ role: "user", content: [{ type: "input_text", text: "Hello" }] },
			{ role: "user", content: [{ type: "input_text", text: "one more thing" }] },
		]);
	});
	for (const buffered of [false, true]) {
		it(`自动续答在附接前按真实收发记账，已缓存结束=${buffered}`, async () => {
			const activityBefore = getModelActivity();
			const rootSessionId = `steer-activity-${buffered}`;
			const owner = { agentId: "Worker", sessionId: rootSessionId, rootSessionId };
			const activity = () => getModelActivity().filter(record => record.rootSessionId === rootSessionId);
			installSocket((frame, socket) => {
				if (frame.type === "response.create") {
					socket.emit(
						{ type: "response.created", response: { id: "activity-first" } },
						...messageFrames("activity-message", "计划"),
					);
				} else {
					socket.emit(
						{
							type: "response.steer.accepted",
							steer: { id: "activity-steer", previous_response_id: "activity-first" },
						},
						{
							type: "response.incomplete",
							response: {
								id: "activity-first",
								status: "incomplete",
								incomplete_details: { reason: "steered" },
								usage: USAGE,
							},
						},
					);
				}
			});
			const model = createGpt6Model();
			const state = new Map<string, ProviderSessionState>();
			const steering = oneShotSteering("继续");
			const user: UserMessage = { role: "user", content: "制定计划", timestamp: Date.now() };
			const first = await withModelActivityOwner(owner, () =>
				streamOpenAICodexResponses(
					model,
					{ systemPrompt: SYSTEM, messages: [user] },
					options(state, steering.source),
				),
			).result();
			expect(steering.settled()).toBe("accepted");
			expect(activity()).toEqual([expect.objectContaining({ ...owner, observable: false })]);
			const socket = ScriptedWebSocket.instances[0]!;
			const duplicate = {
				type: "response.steer.accepted",
				steer: {
					id: "activity-steer",
					previous_response_id: "activity-first",
				},
			};
			socket.emit(duplicate);
			await Promise.resolve();
			expect(activity()).toEqual([expect.objectContaining({ ...owner, observable: false })]);
			withModelActivityOwner({ agentId: "Other", sessionId: "other", rootSessionId: "other" }, () =>
				socket.emit({ type: "response.created", response: { id: "activity-next" } }),
			);
			await Promise.resolve();
			expect(activity()).toEqual([expect.objectContaining({ ...owner, observable: true })]);
			const activeId = activity()[0]!.id;
			socket.emit(duplicate);
			await Promise.resolve();
			expect(
				getModelActivity()
					.map(record => JSON.stringify(record))
					.sort(),
			).toEqual([...activityBefore, ...activity()].map(record => JSON.stringify(record)).sort());
			const complete = () =>
				socket.emit(...messageFrames("activity-next-message", "已继续"), {
					type: "response.completed",
					response: { id: "activity-next", status: "completed", usage: USAGE },
				});
			if (buffered) {
				complete();
				await Promise.resolve();
				expect(activity()).toEqual([]);
			}
			const second = withModelActivityOwner(owner, () =>
				streamOpenAICodexResponses(
					model,
					{
						systemPrompt: SYSTEM,
						messages: [user, first, { role: "user", content: "继续", timestamp: Date.now() }],
					},
					options(state),
				),
			);
			if (!buffered) {
				expect(activity().map(record => record.id)).toEqual([activeId]);
				complete();
			}
			const result = await second.result();
			expect(result.responseId).toBe("activity-next");
			expect(result.content).toEqual([expect.objectContaining({ type: "text", text: "已继续" })]);
			expect(activity()).toEqual([]);
			socket.emit(duplicate);
			await Promise.resolve();
			expect(getModelActivity()).toEqual(activityBefore);
			expect(creates()).toHaveLength(1);
		});
	}

	it("第三轮创建前后收到第一轮旧终态，都不能结算第三轮", async () => {
		const owner = { agentId: "Main", sessionId: "old-frame-three", rootSessionId: "old-frame-three" };
		const activity = () => getModelActivity().filter(record => record.rootSessionId === owner.rootSessionId);
		const thirdSent = Promise.withResolvers<ScriptedWebSocket>();
		const finish = (socket: ScriptedWebSocket, n: number) =>
			socket.emit(...messageFrames(`old-message-${n}`, `答复${n}`), {
				type: "response.completed",
				response: { id: `old-response-${n}`, status: "completed", usage: USAGE },
			});
		installSocket((frame, socket) => {
			if (frame.type !== "response.create") return;
			const n = creates().length;
			if (n === 3) {
				thirdSent.resolve(socket);
				return;
			}
			socket.emit({ type: "response.created", response: { id: `old-response-${n}` } });
			finish(socket, n);
		});
		const model = createGpt6Model();
		const state = new Map<string, ProviderSessionState>();
		const context: Context = { systemPrompt: SYSTEM, messages: [] };
		for (let n = 1; n <= 2; n++) {
			context.messages.push({ role: "user", content: `问题${n}`, timestamp: n });
			const result = await withModelActivityOwner(owner, () =>
				streamOpenAICodexResponses(model, context, options(state)),
			).result();
			expect(result.responseId).toBe(`old-response-${n}`);
			context.messages.push(result);
		}
		context.messages.push({ role: "user", content: "问题3", timestamp: 3 });
		const result = withModelActivityOwner(owner, () =>
			streamOpenAICodexResponses(model, context, options(state)),
		).result();
		const socket = await thirdSent.promise;
		const before = activity();
		expect(before).toEqual([expect.objectContaining({ ...owner, observable: true })]);
		const oldFrame = {
			type: "response.completed",
			response: {
				id: "old-response-1",
				status: "completed",
				usage: USAGE,
			},
		};
		socket.emit(oldFrame);
		await Promise.resolve();
		expect(activity()).toEqual(before);
		socket.emit({ type: "response.created", response: { id: "old-response-3" } }, oldFrame);
		await Promise.resolve();
		expect(activity()).toEqual(before);
		finish(socket, 3);
		const third = await result;
		expect(third.responseId).toBe("old-response-3");
		expect(third.content).toEqual([expect.objectContaining({ type: "text", text: "答复3" })]);
		expect(activity()).toEqual([]);
		expect(ScriptedWebSocket.instances).toHaveLength(1);
	});
});

describe("planSteeredRequest", () => {
	const steer = { role: "user", content: [{ type: "input_text", text: "use tabs" }] };
	const output = { type: "function_call_output", call_id: "call_1", output: "ok" };
	const other = { role: "user", content: [{ type: "input_text", text: "also lint" }] };

	it("attaches when the request is exactly the accepted steering", () => {
		// Responses Lite strips the image `detail` hint from request bodies only.
		const imageSteer = { role: "user", content: [{ type: "input_image", image_url: "u", detail: "auto" }] };
		expect(
			planSteeredRequest([{ role: "user", content: [{ type: "input_image", image_url: "u" }] }], [imageSteer]),
		).toEqual({ kind: "attach" });
	});

	it("sends only the tool output the server awaits", () => {
		expect(planSteeredRequest([output, steer], [steer])).toEqual({ kind: "create", input: [output] });
	});

	it("discards when the request cannot line up with the server queue", () => {
		// Extra input would run alongside the automatic successor.
		expect(planSteeredRequest([steer, other], [steer])).toEqual({ kind: "discard" });
		// The accepted steering is missing from the transcript.
		expect(planSteeredRequest([output], [steer])).toEqual({ kind: "discard" });
		// The chain broke, so the server's continuation point is unusable.
		expect(planSteeredRequest(undefined, [steer])).toEqual({ kind: "discard" });
	});
});
