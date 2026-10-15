import { expect, test } from "bun:test";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { getModelActivity, withModelActivityOwner, withoutModelTransportActivity } from "../src/model-activity";
import { transportFetch } from "../src/utils/transport-fetch";

test("真实HTTP响应头和正文前段不结算模型，认证资料查询不冒充模型", async () => {
	const finish = Promise.withResolvers<void>();
	const encoder = new TextEncoder();
	const server = Bun.serve({
		hostname: "127.0.0.1", port: 0,
		fetch(request) {
			if (new URL(request.url).pathname === "/credentials") return new Response("本机测试凭据");
			return new Response(new ReadableStream<Uint8Array>({
				async start(controller) {
					controller.enqueue(encoder.encode("前段"));
					await finish.promise;
					controller.enqueue(encoder.encode("尾段"));
					controller.close();
				},
			}));
		},
	});
	const rootSessionId = "http-activity-tail";
	const activity = () => getModelActivity().filter(record => record.rootSessionId === rootSessionId);
	const model = { ...getBundledModel("openai", "gpt-4o-mini"), baseUrl: server.url.href };
	const fetchModel = transportFetch(model, undefined);
	try {
		await withModelActivityOwner({ agentId: "Main", sessionId: rootSessionId, rootSessionId }, async () => {
			await withoutModelTransportActivity(async () => {
				const credentials = await fetchModel(new URL("credentials", server.url));
				expect(activity()).toEqual([]);
				expect(await credentials.text()).toBe("本机测试凭据");
			});
			const response = await fetchModel(new URL("inference", server.url));
			expect(activity()).toMatchObject([{ agentId: "Main", observable: true }]);
			const reader = response.body!.getReader();
			expect(new TextDecoder().decode((await reader.read()).value)).toBe("前段");
			expect(activity()).toHaveLength(1);
			finish.resolve();
			let tail = "";
			for (;;) {
				const chunk = await reader.read();
				if (chunk.done) break;
				tail += new TextDecoder().decode(chunk.value);
			}
			expect(tail).toBe("尾段");
			expect(activity()).toEqual([]);
		});
	} finally {
		finish.resolve();
		server.stop(true);
	}
});
