import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import { Socket } from "node:net";
import { resetModelActivityForTests } from "@oh-my-pi/pi-ai/model-activity";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	completeSimple,
	type Context,
	getModelActivity,
	observeModelOperation,
	withModelActivityOwner,
	streamSimple,
	type Usage,
	type UserMessage,
} from "@oh-my-pi/pi-ai";
import { unregisterCustomApis } from "@oh-my-pi/pi-ai/api-registry";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import {
	buildLocalInferenceMessages,
	registerLocalInferenceApi,
} from "@oh-my-pi/pi-coding-agent/tiny/local-inference-api";
import { TINY_LOCAL_MODELS } from "@oh-my-pi/pi-coding-agent/tiny/models";
import { writeJsonLine } from "@oh-my-pi/pi-coding-agent/tiny/jsonl-socket";
import {
	type TinyModelChatOptions,
	TinyTitleClient,
	tinyModelClient,
} from "@oh-my-pi/pi-coding-agent/tiny/title-client";
import type {
	TinyChatMessage,
	TinyWorkerRequest,
	TinyWorkerResponse,
} from "@oh-my-pi/pi-coding-agent/tiny/title-protocol";

afterEach(resetModelActivityForTests);
const SOURCE_ID = "omp/local-inference";
const model = getBundledModel("local", "lfm2.5-230m")!;

function zeroUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function assistantMessage(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "local-inference",
		provider: "local",
		model: model.id,
		usage: zeroUsage(),
		stopReason: "stop",
		timestamp: 2,
	};
}

function userMessage(content: string, timestamp = 1): UserMessage {
	return { role: "user", content, timestamp };
}

async function collectEvents(stream: AsyncIterable<AssistantMessageEvent>): Promise<AssistantMessageEvent[]> {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) events.push(event);
	return events;
}

class FakeTinyWorker {
	terminated = false;
	refCalls = 0;
	unrefCalls = 0;
	#messageHandlers = new Set<(message: TinyWorkerResponse) => void>();
	#errorHandlers = new Set<(error: Error) => void>();
	#onSend: (message: TinyWorkerRequest, worker: FakeTinyWorker) => void;

	constructor(onSend: (message: TinyWorkerRequest, worker: FakeTinyWorker) => void) {
		this.#onSend = onSend;
	}

	send(message: TinyWorkerRequest): boolean {
		this.#onSend(message, this);
		return true;
	}

	onMessage(handler: (message: TinyWorkerResponse) => void): () => void {
		this.#messageHandlers.add(handler);
		return () => this.#messageHandlers.delete(handler);
	}

	onError(handler: (error: Error) => void): () => void {
		this.#errorHandlers.add(handler);
		return () => this.#errorHandlers.delete(handler);
	}

	async terminate(): Promise<void> {
		this.terminated = true;
	}

	ref(): void {
		this.refCalls += 1;
	}

	unref(): void {
		this.unrefCalls += 1;
	}

	emit(message: TinyWorkerResponse): void {
		for (const handler of this.#messageHandlers) handler(message);
	}
}

describe("local inference API", () => {
	beforeEach(() => registerLocalInferenceApi());

	afterEach(() => {
		vi.restoreAllMocks();
		unregisterCustomApis(SOURCE_ID);
	});

	it("flattens assistant context into following user turns and completes through the registered API", async () => {
		let requestedModel: string | undefined;
		let requestedMessages: readonly TinyChatMessage[] | undefined;
		let requestedOptions: TinyModelChatOptions | undefined;
		vi.spyOn(tinyModelClient, "chat").mockImplementation(async (modelKey, messages, options) => {
			requestedModel = modelKey;
			requestedMessages = messages;
			requestedOptions = options;
			return "local answer";
		});
		const context: Context = {
			systemPrompt: ["first system", "second system"],
			messages: [
				userMessage("first user"),
				assistantMessage("assistant context"),
				userMessage("next user", 3),
				assistantMessage("trailing assistant"),
			],
		};

		const result = await completeSimple(model, context, { maxTokens: 37 });

		expect(requestedModel).toBe("lfm2.5-230m");
		expect(requestedMessages).toEqual([
			{ role: "system", content: "first system" },
			{ role: "system", content: "second system" },
			{ role: "user", content: "first user" },
			{ role: "user", content: "assistant context\n\nnext user" },
			{ role: "user", content: "trailing assistant" },
		]);
		expect(requestedOptions).toMatchObject({ maxTokens: 37 });
		expect(result).toMatchObject({
			content: [{ type: "text", text: "local answer" }],
			stopReason: "stop",
			usage: zeroUsage(),
		});
	});

	it("emits the complete assistant text event sequence", async () => {
		vi.spyOn(tinyModelClient, "chat").mockResolvedValue("local answer");
		const stream = streamSimple(model, { messages: [userMessage("question")] });
		const events = await collectEvents(stream);
		const result = await stream.result();

		expect(events.map(event => event.type)).toEqual(["start", "text_start", "text_delta", "text_end", "done"]);
		expect(events[2]).toMatchObject({ type: "text_delta", contentIndex: 0, delta: "local answer" });
		expect(result).toMatchObject({
			content: [{ type: "text", text: "local answer" }],
			stopReason: "stop",
			usage: zeroUsage(),
		});
	});

	it("emits an error instead of a successful empty completion when the worker produces no output", async () => {
		vi.spyOn(tinyModelClient, "chat").mockResolvedValue(null);
		const stream = streamSimple(model, { messages: [userMessage("question")] });
		const events = await collectEvents(stream);
		const result = await stream.result();

		expect(events.map(event => event.type)).toEqual(["start", "error"]);
		expect(events.at(-1)).toMatchObject({
			type: "error",
			reason: "error",
			error: { stopReason: "error", errorMessage: "Local inference returned no output." },
		});
		expect(result).toMatchObject({ stopReason: "error", content: [], usage: zeroUsage() });
	});

	it("keeps cancellation distinct from worker failure", async () => {
		vi.spyOn(tinyModelClient, "chat").mockImplementation(async (_modelKey, _messages, options) => {
			const signal = options?.signal;
			if (!signal) throw new Error("missing test signal");
			if (!signal.aborted) {
				await new Promise<void>(resolve => signal.addEventListener("abort", () => resolve(), { once: true }));
			}
			return null;
		});
		const controller = new AbortController();
		const stream = streamSimple(model, { messages: [userMessage("question")] }, { signal: controller.signal });
		controller.abort(new Error("cancelled locally"));
		const events = await collectEvents(stream);
		const result = await stream.result();

		expect(events.map(event => event.type)).toEqual(["start", "error"]);
		expect(events.at(-1)).toMatchObject({
			type: "error",
			reason: "aborted",
			error: { stopReason: "aborted", errorMessage: "cancelled locally" },
		});
		expect(result.stopReason).toBe("aborted");
	});

	it("rejects non-tiny local models before invoking the worker client", async () => {
		const chat = vi.spyOn(tinyModelClient, "chat").mockResolvedValue("should not run");
		const nonTinyModel = getBundledModel("local", "kokoro")!;
		const stream = streamSimple(nonTinyModel, { messages: [userMessage("question")] });
		const result = await stream.result();

		expect(chat).not.toHaveBeenCalled();
		expect(result).toMatchObject({ stopReason: "error", content: [] });
	});

	it("builds worker messages without dropping text-only structured content", () => {
		expect(
			buildLocalInferenceMessages({
				messages: [
					{ role: "developer", content: [{ type: "text", text: "developer instruction" }], timestamp: 1 },
					{ role: "user", content: [{ type: "text", text: "structured user" }], timestamp: 2 },
				],
			}),
		).toEqual([
			{ role: "system", content: "developer instruction" },
			{ role: "user", content: "structured user" },
		]);
	});
});

describe("tiny model chat client", () => {
	it("accepts every local tiny model key and clamps requested output tokens", async () => {
		const connected: string[] = [];
		const requests: TinyWorkerRequest[] = [];
		const client = new TinyTitleClient(async modelKey => {
			connected.push(modelKey);
			return new FakeTinyWorker((request, worker) => {
				requests.push(request);
				if (request.type === "chat") worker.emit({ type: "text", id: request.id, text: modelKey });
			});
		});

		try {
			for (const [index, spec] of TINY_LOCAL_MODELS.entries()) {
				const maxTokens = index === 0 ? 0 : 100_000;
				expect(await client.chat(spec.key, [{ role: "user", content: "hello" }], { maxTokens })).toBe(spec.key);
			}
			expect(connected).toEqual(TINY_LOCAL_MODELS.map(spec => spec.key));
			expect(requests.map(request => (request.type === "chat" ? request.maxNewTokens : undefined))).toEqual([
				1,
				...TINY_LOCAL_MODELS.slice(1).map(() => 1024),
			]);
		} finally {
			await client.terminate();
		}
	});

	it("取消只结束调用方等待，实际回复前仍保留同一模型请求", async () => {
		const sent = Promise.withResolvers<TinyWorkerRequest>();
		const worker = new FakeTinyWorker(request => sent.resolve(request));
		const client = new TinyTitleClient(async () => worker);
		const controller = new AbortController();
		const rootSessionId = "tiny-cancel-activity";
		const activity = () => getModelActivity().filter(call => call.rootSessionId === rootSessionId);

		try {
			const completion = withModelActivityOwner({ agentId: "Main", sessionId: rootSessionId, rootSessionId }, () =>
				observeModelOperation(() =>
					client.chat("lfm2.5-230m", [{ role: "user", content: "hello" }], {
						signal: controller.signal,
					}),
				),
			);
			const request = await sent.promise;
			const before = activity();
			expect(before).toHaveLength(1);
			controller.abort();
			expect(await completion).toBeNull();
			expect(activity()).toEqual(before);
			worker.emit({ type: "text", id: request.id, text: "迟到结果" });
			expect(activity()).toEqual([]);
		} finally {
			await client.terminate();
		}
	});

	it("prewarms and generates titles with memory-group tiny models", async () => {
		const firstRequest = Promise.withResolvers<TinyWorkerRequest>();
		const worker = new FakeTinyWorker((request, current) => {
			firstRequest.resolve(request);
			if (request.type === "chat")
				current.emit({ type: "text", id: request.id, text: "Memory Model Title</title>" });
		});
		const client = new TinyTitleClient(async () => worker);

		try {
			client.prewarm("qwen3-1.7b");
			expect(await firstRequest.promise).toMatchObject({ type: "ping" });
			expect(await client.generate("qwen3-1.7b", "title this session")).toBe("Memory Model Title");
		} finally {
			await client.terminate();
		}
	});

	it("并发标题保持各自归属，丢失已发送请求的连接只能标记未知", async () => {
		const requests: TinyWorkerRequest[] = [];
		const sent = Promise.withResolvers<void>();
		const worker = new FakeTinyWorker(request => {
			requests.push(request);
			if (requests.length === 2) sent.resolve();
		});
		const client = new TinyTitleClient(async () => worker);
		const rootSessionId = "tiny-title-disconnect";
		const activity = () => getModelActivity().filter(call => call.rootSessionId === rootSessionId);
		const generate = (agentId: string) =>
			withModelActivityOwner({ agentId, sessionId: agentId, rootSessionId }, () =>
				client.generate("lfm2.5-230m", "生成标题"),
			);
		const first = generate("first");
		const second = generate("second");
		try {
			await sent.promise;
			expect(
				activity()
					.map(call => call.agentId)
					.sort(),
			).toEqual(["first", "second"]);
			worker.emit({ type: "text", id: requests[0]!.id, text: "第一个标题</title>" });
			expect(await first).toBe("第一个标题");
			expect(activity().map(call => call.agentId)).toEqual(["second"]);
			await client.terminate();
			expect(await second).toBeNull();
			expect(activity()).toMatchObject([{ agentId: "second", observable: false }]);
		} finally {
			await client.terminate();
		}
	});

	it("连接建立失败且从未发送的请求不遗留模型活动", async () => {
		const rootSessionId = "tiny-connect-failed";
		const client = new TinyTitleClient(async () => {
			throw new Error("本机模拟连接失败");
		});
		try {
			const result = await withModelActivityOwner({ agentId: "Main", sessionId: rootSessionId, rootSessionId }, () =>
				client.generate("lfm2.5-230m", "生成标题"),
			);
			expect(result).toBeNull();
			expect(getModelActivity().filter(call => call.rootSessionId === rootSessionId)).toEqual([]);
		} finally {
			await client.terminate();
		}
	});
	for (const queued of [true, false]) {
		it(`已关闭连接确认未发送时结算活动：${queued ? "连接队列回放" : "已连接句柄"}`, async () => {
			const rootSessionId = `tiny-closed-${queued}`;
			const socket = new Socket();
			socket.destroy();
			const worker = new FakeTinyWorker((request, current) => {
				if (request.type === "chat") current.emit({ type: "text", id: request.id, text: "预先完成" });
			});
			const client = new TinyTitleClient(async () => worker);
			try {
				if (!queued) expect(await client.chat("lfm2.5-230m", [])).toBe("预先完成");
				worker.send = request => writeJsonLine(socket, request);
				const result = await withModelActivityOwner(
					{ agentId: "Main", sessionId: rootSessionId, rootSessionId },
					() => client.chat("lfm2.5-230m", []),
				);
				expect(result).toBeNull();
				expect(getModelActivity().filter(call => call.rootSessionId === rootSessionId)).toEqual([]);
			} finally {
				await client.terminate();
			}
		});
	}

	it("队列回放期间可能已经写出的异常结算调用方但保留未知缺口", async () => {
		const rootSessionId = "tiny-uncertain-send";
		const worker = new FakeTinyWorker(() => {
			throw new Error("无法确认写入是否完成");
		});
		const client = new TinyTitleClient(async () => worker);
		try {
			const result = await withModelActivityOwner({ agentId: "Main", sessionId: rootSessionId, rootSessionId }, () =>
				client.chat("lfm2.5-230m", []),
			);
			expect(result).toBeNull();
			expect(getModelActivity().filter(call => call.rootSessionId === rootSessionId)).toMatchObject([
				{ agentId: "Main", observable: false },
			]);
		} finally {
			await client.terminate();
		}
	});
});
