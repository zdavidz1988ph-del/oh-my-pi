import { resolveExtraCa, withExtraCaInit } from "@oh-my-pi/pi-utils";
import { retainModelActivity, type ModelActivityLease } from "../model-activity";
import { coworkFetch } from "../providers/cowork-fetch";
import { withInferenceUserAgent } from "../providers/inference-headers";
import type { Api, FetchImpl, Model } from "../types";
import { getProxyForProvider, withProxyInit } from "./proxy";
import { createFetchRequestDebugSession, isRequestDebugEnabled } from "./request-debug";

/** Stamped on a fetch already built by {@link transportFetch}. */
const TRANSPORT_FETCH = Symbol("omp.transportFetch");

type TransportFetch = FetchImpl & { [TRANSPORT_FETCH]?: true };

/**
 * The one fetch every inference request goes through. Per call it applies, in
 * order: the inference User-Agent default, `NODE_EXTRA_CA_CERTS`, the
 * per-provider proxy, and `PI_REQ_DEBUG` request/response recording — then
 * calls `fetchImpl` (or the model's default fetch) exactly once. Providers
 * never layer transport concerns themselves.
 *
 * Idempotent: the built fetch is stamped and returned as-is on later passes.
 * `streamSimple` re-enters `stream`, and `streamSimpleRequest` re-enters itself
 * on auth retries, so without the stamp each entry point would add another
 * layer (three PI_REQ_DEBUG dumps for one request).
 */
export function transportFetch(model: Model<Api>, fetchImpl: FetchImpl | undefined): FetchImpl {
	const given = fetchImpl as TransportFetch | undefined;
	if (given?.[TRANSPORT_FETCH]) return given;
	const base =
		given ?? (model.provider === "anthropic" && model.api === "anthropic-messages" ? coworkFetch : globalThis.fetch);
	const proxyUrl = getProxyForProvider(model.provider);

	const fetch: TransportFetch = async (input, init) => {
		init = withInferenceUserAgent(input, init);
		const extraCa = resolveExtraCa();
		if (extraCa) init = withExtraCaInit(init, extraCa);
		if (proxyUrl) init = withProxyInit(input, init, proxyUrl);
		const session = isRequestDebugEnabled() ? await createFetchRequestDebugSession(input, init) : undefined;
		// 等待凭据、名额或调试文件不算发送；响应头到达也不等于传输结束。
		const activity = retainModelActivity();
		let response: Response;
		try { response = await base(input, init); }
		catch (error) { activity.finish(); throw error; }
		try {
			const observed = observeResponse(response, activity);
			return session ? await session.wrapResponse(observed) : observed;
		} catch (error) {
			activity.unobservable();
			throw error;
		}
	};
	if (base.preconnect) fetch.preconnect = base.preconnect;
	fetch[TRANSPORT_FETCH] = true;
	return fetch;
}

function observeResponse(response: Response, activity: ModelActivityLease): Response {
	if (!response.body) {
		activity.finish();
		return response;
	}
	const reader = response.body.getReader();
	const body = new ReadableStream<Uint8Array>({
		async pull(controller) {
			try {
				const { done, value } = await reader.read();
				if (done) {
					activity.finish();
					controller.close();
				} else controller.enqueue(value);
			} catch (error) {
				activity.finish();
				controller.error(error);
			}
		},
		async cancel(reason) {
			try {
				await reader.cancel(reason);
				activity.finish();
			} catch (error) {
				activity.unobservable();
				throw error;
			}
		},
	});
	const wrapped = new Response(body, {
		status: response.status, statusText: response.statusText, headers: response.headers,
	});
	for (const key of ["url", "redirected", "type"] as const) {
		Object.defineProperty(wrapped, key, { value: response[key], configurable: true });
	}
	return wrapped;
}

/** Options-bag form of {@link transportFetch}; returns `options` untouched when its fetch is already built. */
export function withTransportFetch<T extends { fetch?: FetchImpl }>(model: Model<Api>, options: T): T {
	const fetch = transportFetch(model, options.fetch);
	return fetch === options.fetch ? options : { ...options, fetch };
}
