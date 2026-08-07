import type {
	GatewayProviderContext,
	GatewayStreamRequest,
	ModelReasoningOption,
} from "@cline/shared";
import { describe, expect, it } from "vitest";
import { buildAiSdkStreamConfig } from "./ai-sdk";
import {
	resolvePortableReasoning,
	withoutPortableReasoning,
} from "./routing/portable-reasoning";

function request(
	reasoning?: GatewayStreamRequest["reasoning"],
	overrides?: Partial<GatewayStreamRequest>,
): GatewayStreamRequest {
	return {
		providerId: "anthropic",
		modelId: "claude-sonnet-4-6",
		messages: [],
		reasoning,
		...overrides,
	};
}

function context(options?: {
	providerId?: string;
	modelId?: string;
	reasoningOptions?: readonly ModelReasoningOption[];
}): GatewayProviderContext {
	const providerId = options?.providerId ?? "anthropic";
	const modelId = options?.modelId ?? "claude-sonnet-4-6";
	return {
		provider: { id: providerId, name: providerId, models: [] },
		model: {
			id: modelId,
			name: modelId,
			providerId,
			reasoningOptions: options?.reasoningOptions,
		},
		config: { providerId },
	} as GatewayProviderContext;
}

const ALL_EFFORTS: ModelReasoningOption[] = [
	{
		type: "effort",
		values: ["minimal", "low", "medium", "high", "xhigh", "max"],
	},
];

describe("resolvePortableReasoning", () => {
	it.each([
		[{ enabled: false }, "none"],
		[{ effort: "minimal" }, "minimal"],
		[{ effort: "low" }, "low"],
		[{ effort: "medium" }, "medium"],
		[{ effort: "high" }, "high"],
		[{ effort: "xhigh" }, "xhigh"],
		[{ effort: "max" }, "xhigh"],
		[{ enabled: true }, "medium"],
	] as const)("maps %o to %s", (reasoning, expected) => {
		expect(
			resolvePortableReasoning(
				request(reasoning),
				context({ reasoningOptions: ALL_EFFORTS }),
			),
		).toBe(expected);
	});

	it("clamps effort to the catalog-advertised values before mapping", () => {
		expect(
			resolvePortableReasoning(
				request({ effort: "xhigh" }),
				context({
					reasoningOptions: [{ type: "effort", values: ["low", "high"] }],
				}),
			),
		).toBe("high");
		expect(
			resolvePortableReasoning(
				request({ enabled: true }),
				context({
					reasoningOptions: [
						{ type: "effort", values: ["low", "high", "max"] },
					],
				}),
			),
		).toBe("high");
	});

	it("omits reasoning for models with no advertised user-facing control", () => {
		expect(
			resolvePortableReasoning(
				request({ effort: "high" }),
				context({ reasoningOptions: [] }),
			),
		).toBeUndefined();
	});

	it("leaves an exact token budget to provider-specific options", () => {
		expect(
			resolvePortableReasoning(
				request({ enabled: true, effort: "high", budgetTokens: 12_000 }),
				context({
					reasoningOptions: [
						...ALL_EFFORTS,
						{ type: "budget_tokens", min: 1024 },
					],
				}),
			),
		).toBeUndefined();
	});

	it("gives explicit disable precedence over an exact token budget", () => {
		expect(
			resolvePortableReasoning(
				request({ enabled: false, budgetTokens: 12_000 }),
				context(),
			),
		).toBe("none");
	});

	it("gives explicit disable precedence even without an advertised off control", () => {
		expect(
			resolvePortableReasoning(
				request({ enabled: false }),
				context({ reasoningOptions: ALL_EFFORTS }),
			),
		).toBe("none");
	});

	it("leaves explicit disable to native rules for openai-compatible providers", () => {
		const deepseek = request(
			{ enabled: false },
			{ providerId: "deepseek", modelId: "deepseek-v4-flash" },
		);
		const deepseekContext = context({
			providerId: "deepseek",
			modelId: "deepseek-v4-flash",
			reasoningOptions: [
				{ type: "toggle" },
				{ type: "effort", values: ["low", "high", "max"] },
			],
		});
		expect(resolvePortableReasoning(deepseek, deepseekContext)).toBeUndefined();
		// The intent stays on the request so rules like DeepSeek's
		// thinking.type="disabled" remain reachable.
		expect(
			withoutPortableReasoning(deepseek, deepseekContext).reasoning,
		).toEqual({ enabled: false });
	});

	it("removes conflicting controls from native disable requests", () => {
		const normalized = withoutPortableReasoning(
			request(
				{ enabled: false, effort: "high", budgetTokens: 12_000 },
				{ providerId: "custom-provider" },
			),
			context({ providerId: "custom-provider" }),
		);
		expect(normalized.reasoning).toEqual({ enabled: false });
	});

	it("omits reasoning when the caller has no explicit intent", () => {
		expect(resolvePortableReasoning(request(), context())).toBeUndefined();
		expect(resolvePortableReasoning(request({}), context())).toBeUndefined();
	});

	it("adds portable reasoning to supported provider stream settings", () => {
		expect(
			buildAiSdkStreamConfig(
				request({ effort: "high" }),
				context({ reasoningOptions: ALL_EFFORTS }),
			),
		).toMatchObject({ reasoning: "high" });
	});

	it("uses Ollama's top-level reasoning support", () => {
		expect(
			buildAiSdkStreamConfig(
				request({ effort: "high" }, { providerId: "ollama" }),
				context({ providerId: "ollama", modelId: "qwen3:8b" }),
			),
		).toHaveProperty("reasoning", "high");
	});
});
