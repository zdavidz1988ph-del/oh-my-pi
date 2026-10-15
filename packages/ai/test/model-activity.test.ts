import { expect, test } from "bun:test";
import {
	captureModelActivityOwner,
	getModelActivity,
	observeModelOperation,
	observeModelStream,
	withModelActivityOwner,
} from "../src/model-activity";
import { EventStream } from "../src/utils/event-stream";

const rootSessionId = "activity-test-root";
const owner = (agentId: string, sessionId = agentId) => ({ agentId, sessionId, rootSessionId });
const current = () => getModelActivity().filter(call => call.rootSessionId === rootSessionId);
const stream = () =>
	new EventStream<number, number>(
		() => true,
		value => value,
	);

test("同一派工的两个模型独立结算，嵌套入口不重复计数且不抢走事件", async () => {
	const first = stream();
	const second = stream();
	const returned = withModelActivityOwner(owner("child-a"), () =>
		observeModelStream(() => observeModelStream(() => first)),
	);
	withModelActivityOwner(owner("child-b"), () => observeModelStream(() => second));
	try {
		expect(returned).toBe(first);
		expect(
			current()
				.map(call => call.agentId)
				.sort(),
		).toEqual(["child-a", "child-b"]);
		first.push(7);
		await first.result();
		expect(current().map(call => call.agentId)).toEqual(["child-b"]);
		const received: number[] = [];
		for await (const event of returned) received.push(event);
		expect(received).toEqual([7]);
	} finally {
		first.end(7);
		second.end(8);
		await Promise.all([first.result(), second.result()]);
	}
	expect(current()).toEqual([]);
});

test("跨异步工具边界捕获的旧会话不会借用新会话，旧结果不清掉新请求", async () => {
	const old = stream();
	const next = stream();
	const captured = withModelActivityOwner(owner("Main", "old-session"), captureModelActivityOwner);
	await Promise.resolve();
	withModelActivityOwner(owner("Main", "new-session"), () => {
		observeModelStream(() => next);
		withModelActivityOwner(captured, () => observeModelStream(() => old));
	});
	try {
		expect(
			current()
				.map(call => call.sessionId)
				.sort(),
		).toEqual(["new-session", "old-session"]);
		old.end(1);
		await old.result();
		expect(current().map(call => call.sessionId)).toEqual(["new-session"]);
	} finally {
		old.end(1);
		next.end(2);
		await Promise.all([old.result(), next.result()]);
	}
});

test("异步创建和远端压缩实际结算后才清除活动", async () => {
	const creation = Promise.withResolvers<EventStream<number, number>>();
	const remote = Promise.withResolvers<number>();
	const actual = stream();
	const eventual = withModelActivityOwner(owner("create"), () => observeModelStream(() => creation.promise));
	const operation = withModelActivityOwner(owner("compact"), () => observeModelOperation(() => remote.promise));
	try {
		expect(
			current()
				.map(call => call.agentId)
				.sort(),
		).toEqual(["compact", "create"]);
		creation.resolve(actual);
		expect(await eventual).toBe(actual);
		actual.end(3);
		await actual.result();
		expect(current().map(call => call.agentId)).toEqual(["compact"]);
		remote.resolve(9);
		expect(await operation).toBe(9);
		expect(current()).toEqual([]);
	} finally {
		actual.end(0);
		creation.resolve(actual);
		remote.resolve(0);
		await Promise.all([eventual, operation]);
	}
});

test("创建流抛错、异步创建拒绝与远端拒绝都保留原错误并清除活动", async () => {
	const failure = new Error("本机模拟失败");
	expect(() =>
		withModelActivityOwner(owner("sync"), () =>
			observeModelStream(() => {
				throw failure;
			}),
		),
	).toThrow(failure);
	await expect(
		withModelActivityOwner(owner("async"), () => observeModelStream(() => Promise.reject(failure))),
	).rejects.toBe(failure);
	await expect(
		withModelActivityOwner(owner("remote"), () => observeModelOperation(() => Promise.reject(failure))),
	).rejects.toBe(failure);
	expect(current()).toEqual([]);
});

test("重复失联合并身份缺口，跨根超限后保持有界且不假装空闲", () => {
	const source = new URL("../src/model-activity.ts", import.meta.url).href;
	// 进程级未知不能在测试结束时伪造清除；用独立进程避免污染其他用例。
	const result = Bun.spawnSync({
		cmd: [
			process.execPath,
			"--eval",
			`
			import assert from "node:assert/strict";
			import { getModelActivity, retainModelActivity, withModelActivityOwner } from ${JSON.stringify(source)};
			const lose = (root) => withModelActivityOwner({agentId:"Main",sessionId:root,rootSessionId:root}, () => {
				const lease = retainModelActivity();
				lease.unobservable();
				lease.finish();
			});
			for (let i=0;i<1500;i++) lose("same-root");
			assert.equal(getModelActivity().length,1);
			assert.equal(getModelActivity()[0].observable,false);
			assert.equal(getModelActivity()[0].rootSessionId,"same-root");
			for (let i=0;i<700;i++) lose("root-"+i);
			assert.equal(getModelActivity().length,1);
			assert.equal(getModelActivity()[0].observable,false);
			assert.equal(getModelActivity()[0].rootSessionId,undefined);
			const current=withModelActivityOwner({agentId:"Main",sessionId:"new",rootSessionId:"new"},retainModelActivity);
			assert.equal(getModelActivity().filter(x=>x.observable).length,1);
			current.finish();
			assert.equal(getModelActivity().length,1);
			assert.equal(getModelActivity()[0].observable,false);
		`,
		],
		stdout: "pipe",
		stderr: "pipe",
		timeout: 10000,
	});
	expect({ exit: result.exitCode, error: result.stderr.toString() }).toEqual({ exit: 0, error: "" });
});
