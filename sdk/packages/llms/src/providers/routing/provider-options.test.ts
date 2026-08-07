import type {
	GatewayProviderContext,
	GatewayStreamRequest,
	ModelReasoningOption,
} from "@cline/shared";
import { describe, expect, it } from "vitest";
import { BEDROCK_ROUTING_METADATA } from "./bedrock-cache-point";
import { GLM_THINKING_ROUTING_METADATA } from "./glm-thinking";
import { MINIMAX_THINKING_ROUTING_METADATA } from "./minimax-thinking";
import {
	type AiSdkReasoning,
	resolvePortableReasoning,
} from "./portable-reasoning";
import {
	composeAiSdkProviderOptions,
	mergeProviderOptionPatches,
	type ProviderOptionsPatch,
} from "./provider-options";

type RequestOverrides = Partial<GatewayStreamRequest> & {
	providerId: string;
	modelId: string;
};

type ContextOverrides = {
	providerId?: string;
	modelId?: string;
	family?: string;
	contextWindow?: number;
	maxOutputTokens?: number;
	reasoningOptions?: readonly ModelReasoningOption[];
	modelMetadata?: NonNullable<GatewayProviderContext["model"]["metadata"]>;
	capabilities?: GatewayProviderContext["model"]["capabilities"];
	metadata?: GatewayProviderContext["provider"]["metadata"];
	/** Test helper escape hatch for Claude-like models that should not get an auto-injected Anthropic reasoning route. */
	disableAutoAnthropicRouting?: boolean;
};

function makeContext(options?: ContextOverrides): GatewayProviderContext {
	const providerId = options?.providerId ?? "test-provider";
	const modelId = options?.modelId ?? "model-id";
	const normalizedFamily = options?.family?.toLowerCase() ?? "";
	const normalizedModelId = modelId.toLowerCase();
	const useAnthropicReasoningRoute =
		options?.disableAutoAnthropicRouting !== true &&
		(normalizedFamily.includes("claude") ||
			normalizedModelId.includes("claude") ||
			normalizedModelId.includes("anthropic"));
	const anthropicReasoningMetadata: GatewayProviderContext["provider"]["metadata"] =
		useAnthropicReasoningRoute
			? {
					routing: {
						reasoning: {
							format: "anthropic-thinking",
							routes: [
								{
									matcher: "anthropic-compatible",
								},
							],
						},
					},
				}
			: undefined;
	const metadata =
		anthropicReasoningMetadata || options?.metadata
			? {
					...(anthropicReasoningMetadata ?? {}),
					...(options?.metadata ?? {}),
					routing:
						anthropicReasoningMetadata?.routing || options?.metadata?.routing
							? {
									...(anthropicReasoningMetadata?.routing ?? {}),
									...(options?.metadata?.routing ?? {}),
								}
							: undefined,
				}
			: undefined;
	const modelMetadata =
		options?.family || options?.modelMetadata
			? {
					...options.modelMetadata,
					...(options.family ? { family: options.family } : {}),
				}
			: undefined;
	return {
		provider: {
			id: providerId,
			name: providerId,
			defaultModelId: modelId,
			models: [
				{ id: modelId, name: modelId, providerId, capabilities: ["text"] },
			],
			metadata,
		},
		model: {
			id: modelId,
			name: modelId,
			providerId,
			maxOutputTokens: options?.maxOutputTokens,
			contextWindow: options?.contextWindow,
			reasoningOptions: options?.reasoningOptions,
			capabilities: options?.capabilities,
			metadata: modelMetadata,
		},
		config: { providerId },
	};
}

function makeRequest(overrides: RequestOverrides): GatewayStreamRequest {
	return {
		providerId: overrides.providerId,
		modelId: overrides.modelId,
		messages: overrides.messages ?? [
			{
				id: "msg-1",
				role: "user",
				content: [{ type: "text", text: "hi" }],
				createdAt: 0,
			},
		],
		systemPrompt: overrides.systemPrompt,
		temperature: overrides.temperature,
		maxTokens: overrides.maxTokens,
		reasoning: overrides.reasoning,
		signal: overrides.signal,
		tools: overrides.tools,
	};
}

function effortOptions(
	values: Extract<ModelReasoningOption, { type: "effort" }>["values"],
): ModelReasoningOption[] {
	return [{ type: "effort", values }];
}

function budgetOptions(min: number, max?: number): ModelReasoningOption[] {
	return [
		{ type: "budget_tokens", min, ...(max === undefined ? {} : { max }) },
	];
}

/**
 * One row asserts: build a request+context, call composeAiSdkProviderOptions,
 * then for each `expect` entry check that the named bucket either contains or
 * lacks the specified shape. `has` runs through `objectContaining`; `lacks` is
 * a list of property names that must NOT exist in the bucket.
 */
type BucketExpectation = {
	bucket: string;
	has?: Record<string, unknown>;
	lacks?: string[];
};

type Case = {
	name: string;
	request: RequestOverrides;
	context?: ContextOverrides;
	/**
	 * The value the request must resolve to on the AI SDK's portable
	 * top-level reasoning option. Asserted for every case: entries whose
	 * reasoning intent rides the portable channel document it here instead
	 * of a provider-option expectation, and everything else must resolve to
	 * `undefined` so intent cannot silently leak out of provider options.
	 */
	portable?: AiSdkReasoning;
	expect: BucketExpectation[];
};

function runCases(cases: ReadonlyArray<Case>) {
	it.each(cases)("$name", ({
		request,
		context,
		portable,
		expect: expectations,
	}) => {
		const gatewayRequest = makeRequest(request);
		const gatewayContext = makeContext({
			providerId: request.providerId,
			modelId: request.modelId,
			...context,
		});
		const result = composeAiSdkProviderOptions(gatewayRequest, gatewayContext);
		expect(resolvePortableReasoning(gatewayRequest, gatewayContext)).toBe(
			portable,
		);
		for (const e of expectations) {
			const bucket = result[e.bucket];
			if (e.has) {
				expect(bucket).toEqual(expect.objectContaining(e.has));
			}
			if (e.lacks?.length) {
				expect(bucket).toBeDefined();
			}
			for (const key of e.lacks ?? []) {
				expect(bucket).not.toHaveProperty(key);
			}
		}
	});
}

describe("mergeProviderOptionPatches", () => {
	it("returns an empty object when no patches are supplied", () => {
		expect(mergeProviderOptionPatches([])).toEqual({});
	});

	it("ignores undefined patches", () => {
		const patch: ProviderOptionsPatch = { foo: { a: 1 } };
		expect(mergeProviderOptionPatches([undefined, patch, undefined])).toEqual({
			foo: { a: 1 },
		});
	});

	it("merges disjoint buckets", () => {
		expect(
			mergeProviderOptionPatches([{ foo: { a: 1 } }, { bar: { b: 2 } }]),
		).toEqual({ foo: { a: 1 }, bar: { b: 2 } });
	});

	it("later patches win on overlapping keys within the same bucket", () => {
		expect(
			mergeProviderOptionPatches([
				{ foo: { a: 1, b: 2 } },
				{ foo: { b: 99, c: 3 } },
			]),
		).toEqual({ foo: { a: 1, b: 99, c: 3 } });
	});

	it("preserves keys from earlier patches that later patches do not override", () => {
		expect(
			mergeProviderOptionPatches([
				{ foo: { a: 1 } },
				{ foo: { b: 2 } },
				{ foo: { c: 3 } },
			]),
		).toEqual({ foo: { a: 1, b: 2, c: 3 } });
	});
});

describe("composeAiSdkProviderOptions: alias bucket emission", () => {
	it("emits a concrete provider-id bucket and a distinct camelCase alias bucket", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "vercel-ai-gateway",
				modelId: "gpt-5.4",
				reasoning: { effort: "high" },
			}),
			makeContext({ providerId: "vercel-ai-gateway", modelId: "gpt-5.4" }),
		);

		const expected = {};
		expect(result["vercel-ai-gateway"]).toEqual(
			expect.objectContaining({
				...expected,
				strictJsonSchema: false,
			}),
		);
		expect(result.vercelAiGateway).toEqual(
			expect.objectContaining({
				...expected,
				strictJsonSchema: false,
			}),
		);
		expect(result.openaiCompatible).toEqual(
			expect.objectContaining({ strictJsonSchema: false }),
		);
		expect(result["vercel-ai-gateway"]).not.toHaveProperty("effort");
		expect(result["vercel-ai-gateway"]).not.toHaveProperty("reasoningSummary");
	});

	it("disables strict JSON schema for the OpenAI adapter bucket", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({ providerId: "openai-native", modelId: "gpt-5.4" }),
			makeContext({ providerId: "openai-native", modelId: "gpt-5.4" }),
		);

		expect(result.openai).toEqual(
			expect.objectContaining({
				strictJsonSchema: false,
				truncation: "auto",
			}),
		);
		expect(result).not.toHaveProperty("openai-native");
		expect(result).not.toHaveProperty("openaiNative");
		expect(result.openaiCompatible).not.toHaveProperty("strictJsonSchema");
	});

	it("uses the OpenAI adapter bucket for OpenAI Responses compatible providers", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({ providerId: "v0", modelId: "v0-1.5-md" }),
			makeContext({ providerId: "v0", modelId: "v0-1.5-md" }),
			"openai",
		);

		expect(result.openai).toEqual(
			expect.objectContaining({ strictJsonSchema: false }),
		);
		expect(result.openai).not.toHaveProperty("truncation");
		expect(result).not.toHaveProperty("v0");
		expect(result.openaiCompatible).not.toHaveProperty("strictJsonSchema");
	});

	it("does not fan OpenAI-compatible strict schema defaults into native adapter buckets", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({ providerId: "bedrock", modelId: "anthropic.claude-3-5" }),
			makeContext({ providerId: "bedrock", modelId: "anthropic.claude-3-5" }),
		);

		expect(result.bedrock).not.toHaveProperty("strictJsonSchema");
		expect(result.openaiCompatible).not.toHaveProperty("strictJsonSchema");
	});

	it("does not emit anthropic cache_control buckets for bedrock cache-point routing", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "bedrock",
				modelId: "anthropic.claude-sonnet-4-6",
			}),
			makeContext({
				providerId: "bedrock",
				modelId: "anthropic.claude-sonnet-4-6",
				metadata: BEDROCK_ROUTING_METADATA,
			}),
		);

		expect(result.bedrock ?? {}).not.toHaveProperty("cache_control");
		expect(result.anthropic ?? {}).not.toHaveProperty("cache_control");
		expect(result.openaiCompatible ?? {}).not.toHaveProperty("cache_control");
	});

	it("does not emit a separate alias bucket when the alias equals the provider id", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({ providerId: "openai", modelId: "gpt-5" }),
			makeContext({ providerId: "openai", modelId: "gpt-5" }),
		);

		expect(result).toHaveProperty("openai");
		expect(Object.keys(result).filter((k) => k === "openai")).toHaveLength(1);
	});
});

describe("composeAiSdkProviderOptions: Anthropic thinking precedence", () => {
	const MANUAL_THINKING = { type: "enabled", budgetTokens: 1024 };
	const ADAPTIVE_THINKING = { type: "adaptive" };

	runCases([
		{
			name: "Sonnet 4.5 -> manual thinking (no adaptive support), no effort",
			request: {
				providerId: "anthropic",
				modelId: "claude-sonnet-4-5",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				family: "claude-sonnet",
				reasoningOptions: budgetOptions(1024),
			},
			expect: [
				{
					bucket: "anthropic",
					has: { thinking: MANUAL_THINKING },
					lacks: ["effort"],
				},
			],
		},
		{
			// Catalog clamp turns xhigh into max; the portable scale spells
			// max as xhigh and @ai-sdk/anthropic maps it onto the model's
			// strongest supported effort.
			name: "Opus 4.6 clamps unsupported xhigh effort via portable reasoning",
			request: {
				providerId: "anthropic",
				modelId: "claude-opus-4-6",
				reasoning: { enabled: true, effort: "xhigh" },
			},
			context: {
				family: "claude-opus",
				reasoningOptions: effortOptions(["low", "medium", "high", "max"]),
			},
			portable: "xhigh",
			expect: [{ bucket: "anthropic", lacks: ["thinking", "effort"] }],
		},
		{
			name: "Fable 5 routes xhigh effort through portable reasoning",
			request: {
				providerId: "anthropic",
				modelId: "claude-fable-5",
				reasoning: { enabled: true, effort: "xhigh" },
			},
			context: {
				family: "claude-fable",
				reasoningOptions: effortOptions([
					"low",
					"medium",
					"high",
					"xhigh",
					"max",
				]),
			},
			portable: "xhigh",
			expect: [{ bucket: "anthropic", lacks: ["thinking", "effort"] }],
		},
		{
			name: "Sonnet 5 routes explicit disable through portable reasoning",
			request: {
				providerId: "anthropic",
				modelId: "claude-sonnet-5",
				reasoning: { enabled: false },
			},
			context: {
				family: "claude-sonnet",
				reasoningOptions: [
					{ type: "toggle" },
					...effortOptions(["low", "medium", "high", "xhigh", "max"]),
				],
			},
			portable: "none",
			expect: [{ bucket: "anthropic", lacks: ["thinking", "effort"] }],
		},
		{
			// Adaptive-era models reject the manual wire shape even though they
			// also advertise a budget_tokens control, so an explicit numeric
			// budget cannot force manual thinking; the budget is ignored.
			name: "Sonnet 4.6 explicit budget still selects adaptive thinking when effort is advertised",
			request: {
				providerId: "anthropic",
				modelId: "claude-sonnet-4-6",
				reasoning: { enabled: true, budgetTokens: 4096 },
			},
			context: {
				family: "claude-sonnet",
				reasoningOptions: [
					...effortOptions(["low", "medium", "high", "max"]),
					...budgetOptions(1024, 64_000),
				],
			},
			expect: [
				{
					bucket: "anthropic",
					has: { thinking: ADAPTIVE_THINKING },
					lacks: ["effort"],
				},
			],
		},
		{
			name: "budget-only models keep manual thinking for explicit budgets",
			request: {
				providerId: "anthropic",
				modelId: "claude-sonnet-4-5",
				reasoning: { enabled: true, budgetTokens: 4096 },
			},
			context: {
				family: "claude-sonnet",
				reasoningOptions: budgetOptions(1024, 64_000),
			},
			expect: [
				{
					bucket: "anthropic",
					has: { thinking: { type: "enabled", budgetTokens: 4096 } },
					lacks: ["effort"],
				},
			],
		},
		{
			// Unlisted ids still get only broadly supported effort values, so
			// xhigh is downgraded to high before the portable mapping;
			// @ai-sdk/anthropic infers adaptive vs manual thinking itself.
			name: "unknown future Claude aliases without catalog options clamp effort portably",
			request: {
				providerId: "anthropic",
				modelId: "claude-haiku-5",
				reasoning: { enabled: true, effort: "xhigh" },
			},
			context: { family: "claude-haiku" },
			portable: "high",
			expect: [{ bucket: "anthropic", lacks: ["thinking", "effort"] }],
		},
		{
			name: "unlisted adaptive-era suffix variant routes enablement portably",
			request: {
				providerId: "anthropic",
				modelId: "claude-opus-4-6:1m",
				reasoning: { enabled: true },
			},
			context: { family: "claude-opus" },
			portable: "medium",
			expect: [{ bucket: "anthropic", lacks: ["thinking", "effort"] }],
		},
		{
			// @ai-sdk/anthropic derives the manual budget for pre-adaptive
			// ids from the portable level.
			name: "pre-adaptive Claude ids without catalog options route enablement portably",
			request: {
				providerId: "anthropic",
				modelId: "claude-sonnet-4-5-20250929",
				reasoning: { enabled: true },
			},
			context: { family: "claude-sonnet" },
			portable: "medium",
			expect: [{ bucket: "anthropic", lacks: ["thinking", "effort"] }],
		},
		{
			name: "bare enablement without catalog options rides portable reasoning",
			request: {
				providerId: "anthropic",
				modelId: "claude-sonnet-4-5",
				reasoning: { enabled: true },
			},
			context: { family: "claude-sonnet" },
			portable: "medium",
			expect: [{ bucket: "anthropic", lacks: ["thinking", "effort"] }],
		},
		{
			name: "lower non-reasoning Sonnet 3.5 -> no thinking on either bucket",
			request: {
				providerId: "anthropic",
				modelId: "claude-3-5-sonnet-20241022",
				reasoning: { enabled: true, effort: "low" },
			},
			context: { family: "claude-sonnet", capabilities: ["text"] },
			expect: [
				{ bucket: "anthropic", lacks: ["thinking", "effort"] },
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "effort", "reasoning"],
				},
			],
		},
		{
			name: "lower non-reasoning Haiku 3 -> no thinking on either bucket",
			request: {
				providerId: "anthropic",
				modelId: "claude-3-haiku-20240307",
				reasoning: { enabled: true, effort: "low" },
			},
			context: { family: "claude-haiku", capabilities: ["text"] },
			expect: [
				{ bucket: "anthropic", lacks: ["thinking", "effort"] },
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "effort", "reasoning"],
				},
			],
		},
		{
			name: "Cline-routed Sonnet 4.5 -> portable effort, no gateway reasoning object",
			request: {
				providerId: "cline",
				modelId: "anthropic/claude-sonnet-4-5",
				reasoning: { enabled: true, effort: "low" },
			},
			context: { family: "claude-sonnet" },
			portable: "low",
			expect: [{ bucket: "cline", lacks: ["thinking", "reasoning"] }],
		},
		{
			name: "legacy custom Claude with promptCacheStrategy -> portable effort",
			request: {
				providerId: "custom-provider",
				modelId: "anthropic/claude-sonnet-4-5",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				disableAutoAnthropicRouting: true,
				metadata: { promptCacheStrategy: "anthropic-automatic" },
			},
			portable: "high",
			expect: [
				{
					bucket: "custom-provider",
					lacks: [
						"thinking",
						"effort",
						"reasoning",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
				{
					bucket: "openaiCompatible",
					lacks: [
						"thinking",
						"effort",
						"reasoning",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
			],
		},
		{
			name: "unrouted custom Claude -> portable effort",
			request: {
				providerId: "custom-provider",
				modelId: "anthropic/claude-3.5-sonnet",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				disableAutoAnthropicRouting: true,
			},
			portable: "high",
			expect: [
				{
					bucket: "custom-provider",
					lacks: [
						"thinking",
						"effort",
						"reasoning",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
				{
					bucket: "openaiCompatible",
					lacks: [
						"thinking",
						"effort",
						"reasoning",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
			],
		},
	]);

	it.each([
		["provider cap", undefined, 200_000, 128_000],
		["output cap", 2048, 200_000, 2047],
		["small output cap", 64, 4096, 63],
	] as const)("clamps custom Anthropic explicit budgets to the %s", (_, maxTokens, budgetTokens, expected) => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "anthropic",
				modelId: "claude-custom",
				maxTokens,
				reasoning: { enabled: true, budgetTokens },
			}),
			makeContext({
				providerId: "anthropic",
				modelId: "claude-custom",
				family: "claude",
			}),
		);

		expect(result.anthropic).toMatchObject({
			thinking: { type: "enabled", budgetTokens: expected },
		});
		expect(result.openaiCompatible).toMatchObject({
			reasoning: { enabled: true, max_tokens: expected },
		});
	});

	it("does not enable direct Anthropic thinking when disable conflicts with a budget", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "anthropic",
				modelId: "claude-custom",
				reasoning: { enabled: false, budgetTokens: 4096 },
			}),
			makeContext({
				providerId: "anthropic",
				modelId: "claude-custom",
				family: "claude",
			}),
		);

		expect(result.anthropic).not.toHaveProperty("thinking");
	});

	it("drops conflicting Anthropic-compatible budgets after explicit disable", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "custom-provider",
				modelId: "anthropic/claude-custom",
				reasoning: { enabled: false, budgetTokens: 4096 },
			}),
			makeContext({
				providerId: "custom-provider",
				modelId: "anthropic/claude-custom",
				family: "claude",
			}),
		);

		for (const bucket of ["anthropic", "custom-provider", "openaiCompatible"]) {
			expect(result[bucket]).not.toHaveProperty("thinking.type", "enabled");
			expect(result[bucket]).not.toHaveProperty("reasoning.max_tokens");
		}
	});
});

describe("composeAiSdkProviderOptions: family/provider thinking patches", () => {
	runCases([
		{
			name: "openrouter reasoning budgetTokens -> reasoning.max_tokens",
			request: {
				providerId: "openrouter",
				modelId: "openai/gpt-oss-120b",
				reasoning: { budgetTokens: 1024 },
			},
			expect: [
				{
					bucket: "openrouter",
					has: { reasoning: { max_tokens: 1024 } },
					lacks: ["thinking", "effort", "reasoningEffort"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		{
			// The portable scale spells the advertised "max" as xhigh.
			name: "openrouter routes catalog maximum effort through portable reasoning",
			request: {
				providerId: "openrouter",
				modelId: "moonshotai/reasoning-model",
				reasoning: { effort: "max" },
			},
			context: {
				reasoningOptions: effortOptions(["low", "medium", "high", "max"]),
			},
			portable: "xhigh",
			expect: [
				{
					bucket: "openrouter",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		{
			name: "openrouter routes supported Anthropic xhigh effort portably",
			request: {
				providerId: "openrouter",
				modelId: "anthropic/claude-opus-4-7",
				reasoning: { effort: "xhigh" },
			},
			context: {
				family: "claude-opus",
				reasoningOptions: effortOptions([
					"low",
					"medium",
					"high",
					"xhigh",
					"max",
				]),
			},
			portable: "xhigh",
			expect: [
				{
					bucket: "openrouter",
					lacks: ["thinking", "reasoning", "reasoningEffort"],
				},
			],
		},
		{
			name: "openrouter reasoning enabled-only rides portable reasoning",
			request: {
				providerId: "openrouter",
				modelId: "openai/gpt-oss-120b",
				reasoning: { enabled: true },
			},
			portable: "medium",
			expect: [
				{
					bucket: "openrouter",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		{
			name: "models.dev default effort bypasses unlisted-model heuristics",
			request: {
				providerId: "groq",
				modelId: "qwen/qwen3-32b",
				reasoning: { enabled: true },
			},
			context: {
				family: "qwen",
				reasoningOptions: effortOptions(["none", "default"]),
			},
			expect: [
				{
					bucket: "groq",
					has: { reasoningEffort: "default" },
					lacks: ["effort", "reasoningSummary"],
				},
			],
		},
		{
			// The models.dev "default" effort has no portable equivalent and no
			// Anthropic wire shape; the model's own default thinking applies.
			name: "Anthropic default effort leaves the model default in charge",
			request: {
				providerId: "anthropic",
				modelId: "claude-future",
				reasoning: { enabled: true },
			},
			context: {
				family: "claude",
				reasoningOptions: effortOptions(["none", "default"]),
			},
			expect: [
				{
					bucket: "anthropic",
					lacks: ["thinking", "effort", "reasoningEffort"],
				},
			],
		},
		{
			name: "openrouter reasoning enabled-only with a request cap rides portable reasoning",
			request: {
				providerId: "openrouter",
				modelId: "openai/gpt-oss-120b",
				maxTokens: 10_000,
				reasoning: { enabled: true },
			},
			portable: "medium",
			expect: [
				{
					bucket: "openrouter",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		{
			name: "openrouter reasoning enabled-only with a model cap rides portable reasoning",
			request: {
				providerId: "openrouter",
				modelId: "openai/gpt-oss-120b",
				reasoning: { enabled: true },
			},
			context: { maxOutputTokens: 12_000 },
			portable: "medium",
			expect: [
				{
					bucket: "openrouter",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		{
			name: "openrouter reasoning effort rides portable reasoning",
			request: {
				providerId: "openrouter",
				modelId: "openai/gpt-oss-120b",
				reasoning: { effort: "high" },
			},
			portable: "high",
			expect: [
				{
					bucket: "openrouter",
					lacks: ["thinking", "reasoning", "reasoningEffort"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		{
			name: "openrouter unset reasoning -> no reasoning field",
			request: {
				providerId: "openrouter",
				modelId: "deepseek/deepseek-v4-pro",
			},
			expect: [
				{
					bucket: "openrouter",
					lacks: ["reasoning", "thinking", "effort", "reasoningEffort"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		{
			name: "openrouter Qwen family -> prompt cache buckets without Anthropic thinking",
			request: {
				providerId: "openrouter",
				modelId: "alibaba/qwen3.6-plus",
			},
			context: {
				family: "qwen",
				metadata: { promptCacheStrategy: "anthropic-automatic" },
			},
			expect: [
				{
					bucket: "openrouter",
					has: { cache_control: { type: "ephemeral" } },
					lacks: ["thinking", "effort", "reasoning"],
				},
				{
					bucket: "openaiCompatible",
					has: { cache_control: { type: "ephemeral" } },
					lacks: ["thinking", "effort", "reasoning"],
				},
			],
		},
		// GLM/Z.AI routed reasoning — enabled
		{
			name: "openrouter GLM thinking-enabled rides portable reasoning, no thinking leak",
			request: {
				providerId: "openrouter",
				modelId: "z-ai/glm-4.7",
				reasoning: { enabled: true },
			},
			portable: "medium",
			expect: [
				{
					bucket: "openrouter",
					lacks: ["thinking", "reasoning"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning"],
				},
			],
		},
		{
			name: "openrouter GLM budgetTokens -> OpenRouter max_tokens is not overwritten by routed GLM",
			request: {
				providerId: "openrouter",
				modelId: "z-ai/glm-4.7",
				reasoning: { enabled: true, budgetTokens: 1024 },
			},
			expect: [
				{
					bucket: "openrouter",
					has: { reasoning: { max_tokens: 1024 } },
					lacks: ["thinking"],
				},
				{
					bucket: "openaiCompatible",
					has: { reasoning: { enabled: true } },
					lacks: ["thinking"],
				},
			],
		},
		{
			name: "openrouter GLM effort rides portable reasoning",
			request: {
				providerId: "openrouter",
				modelId: "z-ai/glm-4.7",
				reasoning: { enabled: true, effort: "medium" },
			},
			portable: "medium",
			expect: [
				{
					bucket: "openrouter",
					lacks: ["thinking", "reasoning"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning"],
				},
			],
		},
		{
			name: "vercel-ai-gateway GLM thinking-enabled rides portable reasoning, no thinking leak",
			request: {
				providerId: "vercel-ai-gateway",
				modelId: "z-ai/glm-4.7",
				reasoning: { enabled: true },
			},
			portable: "medium",
			expect: [
				{
					bucket: "vercel-ai-gateway",
					lacks: ["thinking", "reasoning"],
				},
				{
					bucket: "vercelAiGateway",
					lacks: ["thinking", "reasoning"],
				},
				{ bucket: "openaiCompatible", lacks: ["thinking", "reasoning"] },
			],
		},
		// GLM/Z.AI routed reasoning — disabled
		{
			name: "openrouter GLM thinking-disabled -> reasoning.effort=none in provider+compatible",
			request: {
				providerId: "openrouter",
				modelId: "z-ai/glm-4.7",
				reasoning: { enabled: false },
			},
			expect: [
				{ bucket: "openrouter", has: { reasoning: { effort: "none" } } },
				// The OpenRouter bucket is authoritative on the wire; this residual
				// compatible bucket remains for non-OpenRouter routed GLM paths.
				{ bucket: "openaiCompatible", has: { reasoning: { exclude: true } } },
			],
		},
		{
			name: "vercel-ai-gateway GLM thinking-disabled -> reasoning.exclude in provider+alias",
			request: {
				providerId: "vercel-ai-gateway",
				modelId: "z-ai/glm-4.7",
				reasoning: { enabled: false },
			},
			expect: [
				{ bucket: "vercel-ai-gateway", has: { reasoning: { exclude: true } } },
				{ bucket: "vercelAiGateway", has: { reasoning: { exclude: true } } },
			],
		},
		{
			name: "cline GLM thinking-disabled -> routed reasoning only, no thinking leak",
			request: {
				providerId: "cline",
				modelId: "z-ai/glm-4.7",
				reasoning: { enabled: false },
			},
			expect: [
				{
					bucket: "cline",
					has: { reasoning: { exclude: true } },
					lacks: ["thinking"],
				},
				{
					bucket: "openaiCompatible",
					has: { reasoning: { exclude: true } },
					lacks: ["thinking"],
				},
			],
		},
		// Native Z.AI enablement rides portable reasoning; explicit disable
		// keeps the native thinking shape (see the disable cases).
		{
			name: "native zai thinking-enabled rides portable reasoning",
			request: {
				providerId: "zai",
				modelId: "glm-4.7",
				reasoning: { enabled: true },
			},
			context: { family: "glm", metadata: GLM_THINKING_ROUTING_METADATA },
			portable: "medium",
			expect: [
				{ bucket: "zai", lacks: ["thinking", "reasoning"] },
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning"],
				},
			],
		},
		{
			name: "native zai custom non-GLM -> no generic adaptive thinking",
			request: {
				providerId: "zai",
				modelId: "zai-other-model",
				reasoning: { enabled: true },
			},
			context: { family: "other", metadata: GLM_THINKING_ROUTING_METADATA },
			portable: "medium",
			expect: [
				{ bucket: "zai", lacks: ["thinking", "reasoning"] },
				{ bucket: "openaiCompatible", lacks: ["thinking", "reasoning"] },
			],
		},
		// Kimi K2.6 family: explicit enabled/disabled and unset defaults to enabled
		{
			name: "cline Kimi K2.6 family reasoning.enabled=false -> thinking.type=disabled",
			request: {
				providerId: "cline",
				modelId: "moonshotai/kimi-k2.6",
				reasoning: { enabled: false },
			},
			context: { family: "kimi-k2.6" },
			expect: [
				{ bucket: "cline", has: { thinking: { type: "disabled" } } },
				{
					bucket: "openaiCompatible",
					has: { thinking: { type: "disabled" } },
				},
			],
		},
		{
			// K2.6 thinks by default; explicit enablement rides portable
			// reasoning instead of the native thinking toggle.
			name: "cline Kimi K2.6 family reasoning.enabled=true rides portable reasoning",
			request: {
				providerId: "cline",
				modelId: "moonshotai/kimi-k2.6",
				reasoning: { enabled: true },
			},
			context: { family: "kimi-k2.6" },
			portable: "medium",
			expect: [
				{ bucket: "cline", lacks: ["thinking"] },
				{ bucket: "openaiCompatible", lacks: ["thinking"] },
			],
		},
		{
			name: "cline generic reasoning.enabled=false -> gateway reasoning only, no thinking patch",
			request: {
				providerId: "cline",
				modelId: "gpt-5.4",
				reasoning: { enabled: false },
			},
			expect: [
				{
					bucket: "cline",
					has: { reasoning: { enabled: false } },
					lacks: ["thinking"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning"],
				},
			],
		},
		{
			name: "cline non-K2.6 Moonshot Kimi reasoning.enabled=false -> thinking.type=disabled",
			request: {
				providerId: "cline",
				modelId: "moonshotai/kimi-k2.5",
				reasoning: { enabled: false },
			},
			context: { family: "kimi-k2.5" },
			expect: [
				{
					bucket: "cline",
					has: {
						reasoning: { enabled: false },
						thinking: { type: "disabled" },
					},
				},
				{
					bucket: "openaiCompatible",
					has: { thinking: { type: "disabled" } },
				},
			],
		},
		{
			name: "openrouter Kimi K2.6 family reasoning.enabled=false -> reasoning.effort=none",
			request: {
				providerId: "openrouter",
				modelId: "moonshotai/kimi-k2.6",
				reasoning: { enabled: false },
			},
			context: { family: "kimi-k2.6" },
			expect: [
				{
					bucket: "openrouter",
					has: { reasoning: { effort: "none" } },
					lacks: ["thinking"],
				},
				{ bucket: "openaiCompatible", lacks: ["thinking"] },
			],
		},
		{
			// No explicit intent leaves K2.6's default-on thinking untouched.
			name: "openai-compatible Kimi K2.6 family unset reasoning -> no thinking toggle",
			request: { providerId: "openai-compatible", modelId: "kimi-k2.6" },
			context: { family: "kimi-k2.6" },
			expect: [
				{
					bucket: "openai-compatible",
					lacks: ["thinking"],
				},
				{ bucket: "openaiCompatible", lacks: ["thinking"] },
			],
		},
		{
			name: "openai-compatible Kimi K2.6 family reasoning.enabled=false -> thinking.type=disabled",
			request: {
				providerId: "openai-compatible",
				modelId: "kimi-k2.6",
				reasoning: { enabled: false },
			},
			context: { family: "kimi-k2.6" },
			expect: [
				{
					bucket: "openai-compatible",
					has: { thinking: { type: "disabled" } },
				},
				{
					bucket: "openaiCompatible",
					has: { thinking: { type: "disabled" } },
				},
			],
		},
		{
			name: "openai-compatible Kimi K2.6 family empty reasoning -> no thinking toggle",
			request: {
				providerId: "openai-compatible",
				modelId: "kimi-k2.6",
				reasoning: {},
			},
			context: { family: "kimi-k2.6" },
			expect: [{ bucket: "openaiCompatible", lacks: ["thinking"] }],
		},
		{
			name: "qwen prompt-cache-only route reasoning.enabled=true -> cache control, no thinking",
			request: {
				providerId: "qwen",
				modelId: "qwen-plus",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				family: "qwen",
				capabilities: ["text", "prompt-cache"],
				metadata: {
					routing: {
						promptCache: {
							format: "anthropic-cache-control",
							routes: [
								{
									matcher: "model-family",
									family: "qwen",
									requiredCapability: "prompt-cache",
								},
							],
						},
					},
				},
			},
			expect: [
				{
					bucket: "qwen",
					has: { cache_control: { type: "ephemeral" } },
					lacks: ["thinking", "effort", "reasoningEffort", "reasoningSummary"],
				},
				{
					bucket: "openaiCompatible",
					has: { cache_control: { type: "ephemeral" } },
					lacks: ["thinking", "effort", "reasoningEffort", "reasoningSummary"],
				},
			],
		},
		{
			name: "qwen prompt-cache route without capability reasoning.enabled=true -> no generic thinking",
			request: {
				providerId: "qwen",
				modelId: "qwen-plus",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				family: "qwen",
				capabilities: ["text"],
				metadata: {
					routing: {
						promptCache: {
							format: "anthropic-cache-control",
							routes: [
								{
									matcher: "model-family",
									family: "qwen",
									requiredCapability: "prompt-cache",
								},
							],
						},
					},
				},
			},
			expect: [
				{
					bucket: "qwen",
					lacks: [
						"cache_control",
						"thinking",
						"effort",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
				{
					bucket: "openaiCompatible",
					lacks: [
						"cache_control",
						"thinking",
						"effort",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
			],
		},
		{
			name: "cline qwen prompt-cache-only route reasoning.enabled=true -> cache control, no gateway reasoning",
			request: {
				providerId: "cline",
				modelId: "qwen/qwen3.6-plus",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				family: "qwen",
				capabilities: ["text", "prompt-cache"],
				metadata: {
					routing: {
						promptCache: {
							format: "anthropic-cache-control",
							routes: [
								{
									matcher: "model-family",
									family: "qwen",
									requiredCapability: "prompt-cache",
								},
							],
						},
					},
				},
			},
			expect: [
				{
					bucket: "cline",
					has: { cache_control: { type: "ephemeral" } },
					lacks: [
						"reasoning",
						"thinking",
						"effort",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
			],
		},
		{
			portable: "high",
			name: "cline unregistered qwen reasoning.enabled=true -> no gateway reasoning",
			request: {
				providerId: "cline",
				modelId: "qwen/qwen3.7-plus",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				metadata: {
					routing: {
						promptCache: {
							format: "anthropic-cache-control",
							routes: [
								{
									matcher: "model-family",
									family: "qwen",
									requiredCapability: "prompt-cache",
								},
							],
						},
					},
				},
			},
			expect: [
				{
					bucket: "cline",
					lacks: [
						"reasoning",
						"thinking",
						"effort",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "effort", "reasoningEffort", "reasoningSummary"],
				},
			],
		},
		{
			name: "cline Kimi K2.6 family reasoning.enabled=false also keeps gateway reasoning shape",
			request: {
				providerId: "cline",
				modelId: "moonshotai/kimi-k2.6",
				reasoning: { enabled: false },
			},
			context: { family: "kimi-k2.6" },
			expect: [{ bucket: "cline", has: { reasoning: { enabled: false } } }],
		},
		{
			name: "cline Claude Fable omits an unadvertised disabled control",
			request: {
				providerId: "cline",
				modelId: "anthropic/claude-fable-5",
				reasoning: { enabled: false },
			},
			context: {
				reasoningOptions: effortOptions([
					"low",
					"medium",
					"high",
					"xhigh",
					"max",
				]),
			},
			expect: [
				{
					bucket: "cline",
					lacks: ["reasoning", "thinking"],
				},
			],
		},
		{
			name: "cline Claude Fable ignores an advertised toggle because reasoning is mandatory",
			request: {
				providerId: "cline",
				modelId: "anthropic/claude-fable-5",
				reasoning: { enabled: false },
			},
			context: {
				reasoningOptions: [
					{ type: "toggle" },
					...effortOptions(["low", "medium", "high", "xhigh", "max"]),
				],
			},
			expect: [
				{
					bucket: "cline",
					lacks: ["reasoning", "thinking"],
				},
			],
		},
		{
			name: "cline StepFun 3.7 Flash reasoning.enabled=false omits disabled reasoning",
			request: {
				providerId: "cline",
				modelId: "stepfun/step-3.7-flash",
				reasoning: { enabled: false },
			},
			expect: [
				{
					bucket: "cline",
					lacks: ["reasoning", "thinking"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["reasoning", "thinking"],
				},
			],
		},
		{
			name: "cline StepFun 3.7 Flash variants reasoning.enabled=false omit disabled reasoning",
			request: {
				providerId: "cline",
				modelId: "stepfun/step-3.7-flash-v2",
				reasoning: { enabled: false },
			},
			expect: [
				{
					bucket: "cline",
					lacks: ["reasoning", "thinking"],
				},
			],
		},
		// OpenRouter owns the reasoning object regardless of Moonshot family.
		{
			name: "openrouter non-K2.6 Moonshot Kimi reasoning.enabled=false -> reasoning.effort=none",
			request: {
				providerId: "openrouter",
				modelId: "moonshotai/kimi-k2.5",
				reasoning: { enabled: false },
			},
			expect: [
				{
					bucket: "openrouter",
					has: { reasoning: { effort: "none" } },
					lacks: ["thinking"],
				},
				{ bucket: "openaiCompatible", lacks: ["thinking"] },
			],
		},
		// DeepSeek family — direct provider id and openai-compatible via family
		{
			name: "direct deepseek reasoning disable -> thinking.type=disabled",
			request: {
				providerId: "deepseek",
				modelId: "deepseek-v4-pro",
				reasoning: { enabled: false },
			},
			expect: [
				{ bucket: "deepseek", has: { thinking: { type: "disabled" } } },
				{
					bucket: "openaiCompatible",
					has: { thinking: { type: "disabled" } },
				},
			],
		},
		{
			name: "direct deepseek reasoning enable rides portable reasoning",
			request: {
				providerId: "deepseek",
				modelId: "deepseek-v4-pro",
				reasoning: { enabled: true },
			},
			portable: "medium",
			expect: [
				{ bucket: "deepseek", lacks: ["thinking"] },
				{ bucket: "openaiCompatible", lacks: ["thinking"] },
			],
		},
		{
			name: "openai-compatible deepseek family reasoning.enabled=false -> thinking.type=disabled",
			request: {
				providerId: "openai-compatible",
				modelId: "deepseek-v4-pro",
				reasoning: { enabled: false },
			},
			context: { family: "deepseek" },
			expect: [
				{
					bucket: "openai-compatible",
					has: { thinking: { type: "disabled" } },
				},
				{
					bucket: "openaiCompatible",
					has: { thinking: { type: "disabled" } },
				},
			],
		},
		{
			name: "openai-compatible deepseek-thinking family reasoning.enabled=false -> thinking.type=disabled",
			request: {
				providerId: "openai-compatible",
				modelId: "deepseek-v4-pro",
				reasoning: { enabled: false },
			},
			context: { family: "deepseek-thinking" },
			expect: [
				{
					bucket: "openai-compatible",
					has: { thinking: { type: "disabled" } },
				},
				{
					bucket: "openaiCompatible",
					has: { thinking: { type: "disabled" } },
				},
			],
		},
		{
			name: "openai-compatible deepseek-flash family reasoning.enabled=false -> thinking.type=disabled",
			request: {
				providerId: "openai-compatible",
				modelId: "deepseek-v4-pro",
				reasoning: { enabled: false },
			},
			context: { family: "deepseek-flash" },
			expect: [
				{
					bucket: "openai-compatible",
					has: { thinking: { type: "disabled" } },
				},
				{
					bucket: "openaiCompatible",
					has: { thinking: { type: "disabled" } },
				},
			],
		},
		{
			name: "openai-compatible deepseek family reasoning.enabled=true rides portable reasoning",
			request: {
				providerId: "openai-compatible",
				modelId: "deepseek-v4-pro",
				reasoning: { enabled: true },
			},
			context: { family: "deepseek-thinking" },
			portable: "medium",
			expect: [
				{ bucket: "openai-compatible", lacks: ["thinking"] },
				{ bucket: "openaiCompatible", lacks: ["thinking"] },
			],
		},
		{
			name: "openai-compatible deepseek family with unset reasoning -> no thinking emitted",
			request: { providerId: "openai-compatible", modelId: "deepseek-v4-pro" },
			context: { family: "deepseek" },
			expect: [
				{ bucket: "openai-compatible", lacks: ["thinking"] },
				{ bucket: "openaiCompatible", lacks: ["thinking"] },
			],
		},
		{
			name: "openrouter MiniMax M3 reasoning enabled rides portable reasoning",
			request: {
				providerId: "openrouter",
				modelId: "minimax/minimax-m3",
				reasoning: { enabled: true },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
			},
			portable: "medium",
			expect: [
				{
					bucket: "openrouter",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		{
			name: "openrouter MiniMax M3 reasoning disabled -> OpenRouter reasoning.effort=none",
			request: {
				providerId: "openrouter",
				modelId: "minimax/minimax-m3",
				reasoning: { enabled: false },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
			},
			expect: [
				{
					bucket: "openrouter",
					has: { reasoning: { effort: "none" } },
					lacks: ["thinking"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning"],
				},
			],
		},
		{
			name: "vercel MiniMax M3 reasoning enabled rides portable reasoning",
			request: {
				providerId: "vercel-ai-gateway",
				modelId: "minimax/minimax-m3",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
			},
			portable: "high",
			expect: [
				{
					bucket: "vercel-ai-gateway",
					lacks: [
						"thinking",
						"reasoning",
						"effort",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
				{
					bucket: "vercelAiGateway",
					lacks: [
						"thinking",
						"reasoning",
						"effort",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
			],
		},
		{
			name: "vercel MiniMax M3 reasoning disabled -> gateway reasoning.exclude",
			request: {
				providerId: "vercel-ai-gateway",
				modelId: "minimax/minimax-m3",
				reasoning: { enabled: false },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
			},
			expect: [
				{
					bucket: "vercel-ai-gateway",
					has: { reasoning: { exclude: true } },
					lacks: ["thinking"],
				},
				{
					bucket: "vercelAiGateway",
					has: { reasoning: { exclude: true } },
					lacks: ["thinking"],
				},
			],
		},
		{
			name: "vercel MiniMax M3 sibling without advertised controls -> no reasoning control",
			request: {
				providerId: "vercel-ai-gateway",
				modelId: "minimax/minimax-m3-pro",
				reasoning: { enabled: true },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
			},
			portable: "medium",
			expect: [
				{
					bucket: "vercel-ai-gateway",
					lacks: ["thinking", "reasoning"],
				},
			],
		},
		{
			name: "cline MiniMax M3 reasoning enabled rides portable reasoning without thinking leak",
			request: {
				providerId: "cline",
				modelId: "minimax/minimax-m3",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
			},
			portable: "high",
			expect: [
				{
					bucket: "cline",
					lacks: [
						"thinking",
						"reasoning",
						"effort",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		{
			name: "cline MiniMax M3 reasoning disabled -> gateway reasoning disabled",
			request: {
				providerId: "cline",
				modelId: "minimax/minimax-m3",
				reasoning: { enabled: false },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
			},
			expect: [
				{
					bucket: "cline",
					has: { reasoning: { enabled: false } },
					lacks: ["thinking"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning"],
				},
			],
		},
		{
			name: "direct MiniMax M3 reasoning disabled -> thinking.type=disabled",
			request: {
				providerId: "minimax",
				modelId: "MiniMax-M3",
				reasoning: { enabled: false },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
				metadata: MINIMAX_THINKING_ROUTING_METADATA,
			},
			expect: [
				{
					bucket: "minimax",
					has: { thinking: { type: "disabled" } },
					lacks: ["reasoning", "effort", "reasoningEffort", "reasoningSummary"],
				},
				{
					bucket: "openaiCompatible",
					has: { thinking: { type: "disabled" } },
					lacks: ["reasoning", "effort", "reasoningEffort", "reasoningSummary"],
				},
			],
		},
		{
			name: "direct MiniMax M3 reasoning enabled rides portable reasoning",
			request: {
				providerId: "minimax",
				modelId: "MiniMax-M3",
				reasoning: { enabled: true, effort: "high" },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
				metadata: MINIMAX_THINKING_ROUTING_METADATA,
			},
			portable: "high",
			expect: [
				{
					bucket: "minimax",
					lacks: [
						"thinking",
						"reasoning",
						"effort",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
				{
					bucket: "openaiCompatible",
					lacks: [
						"thinking",
						"reasoning",
						"effort",
						"reasoningEffort",
						"reasoningSummary",
					],
				},
			],
		},
		{
			name: "direct MiniMax M2.5 without an advertised control -> no generic thinking",
			request: {
				providerId: "minimax",
				modelId: "MiniMax-M2.5",
				reasoning: { enabled: true },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
				metadata: MINIMAX_THINKING_ROUTING_METADATA,
			},
			portable: "medium",
			expect: [
				{
					bucket: "minimax",
					lacks: ["thinking", "reasoning"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning"],
				},
			],
		},
		{
			name: "direct MiniMax M2.7 reasoning disabled -> no MiniMax M3 disabled exception",
			request: {
				providerId: "minimax",
				modelId: "MiniMax-M2.7",
				reasoning: { enabled: false },
			},
			context: {
				family: "minimax",
				capabilities: ["reasoning"],
				metadata: MINIMAX_THINKING_ROUTING_METADATA,
			},
			expect: [
				{
					bucket: "minimax",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["thinking", "reasoning", "effort", "reasoningEffort"],
				},
			],
		},
		// Ollama: explicit intent rides the portable option (the patched
		// ollama-ai-provider-v2 maps it onto the native `think` flag; see
		// vendors/ollama.wire.test.ts). Unset reasoning sends nothing so the
		// Ollama server default (auto-thinking for capable models) applies.
		{
			name: "ollama metadata reasoningDefaultOn disabled -> portable none",
			request: {
				providerId: "ollama",
				modelId: "local-known-reasoner:latest",
				reasoning: { enabled: false },
			},
			context: { modelMetadata: { reasoningDefaultOn: true } },
			portable: "none",
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 32768 } },
					lacks: ["think"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["think", "reasoningEffort", "reasoning"],
				},
			],
		},
		{
			name: "ollama qwen3 fallback reasoning disabled -> portable none",
			request: {
				providerId: "ollama",
				modelId: "qwen3-coder:30b",
				reasoning: { enabled: false },
			},
			portable: "none",
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 32768 } },
					lacks: ["think"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["think", "reasoningEffort", "reasoning"],
				},
			],
		},
		{
			name: "ollama qwen3 fallback reasoning enabled -> portable medium",
			request: {
				providerId: "ollama",
				modelId: "qwen3-coder:30b",
				reasoning: { enabled: true },
			},
			portable: "medium",
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 32768 } },
					lacks: ["think", "reasoningEffort", "reasoning"],
				},
				{ bucket: "openaiCompatible", lacks: ["reasoningEffort", "reasoning"] },
			],
		},
		{
			name: "ollama qwen3 fallback with unset reasoning leaves the server default",
			request: {
				providerId: "ollama",
				modelId: "qwen3-coder:30b",
			},
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 32768 } },
					lacks: ["think", "reasoningEffort", "reasoning"],
				},
				{ bucket: "openaiCompatible", lacks: ["reasoningEffort", "reasoning"] },
			],
		},
		{
			name: "ollama metadata reasoningDefaultOn with unset reasoning leaves the server default",
			request: {
				providerId: "ollama",
				modelId: "local-known-reasoner:latest",
			},
			context: { modelMetadata: { reasoningDefaultOn: true } },
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 32768 } },
					lacks: ["think", "reasoningEffort", "reasoning"],
				},
				{ bucket: "openaiCompatible", lacks: ["reasoningEffort", "reasoning"] },
			],
		},
		{
			name: "ollama deepseek family disable rides portable none, no thinking leak",
			request: {
				providerId: "ollama",
				modelId: "deepseek-r1:latest",
				reasoning: { enabled: false },
			},
			context: {
				family: "deepseek",
				modelMetadata: { reasoningDefaultOn: true },
			},
			portable: "none",
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 32768 } },
					lacks: ["think", "thinking"],
				},
				{
					bucket: "openaiCompatible",
					lacks: ["think", "thinking", "reasoningEffort", "reasoning"],
				},
			],
		},
		{
			name: "ollama explicit disable overrides metadata reasoningDefaultOn false",
			request: {
				providerId: "ollama",
				modelId: "qwen3-coder:30b",
				reasoning: { enabled: false },
			},
			context: { modelMetadata: { reasoningDefaultOn: false } },
			portable: "none",
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 32768 } },
					lacks: ["think", "reasoningEffort", "reasoning"],
				},
				{ bucket: "openaiCompatible", lacks: ["reasoningEffort", "reasoning"] },
			],
		},
		{
			name: "ollama local model explicit reasoning disabled -> portable none",
			request: {
				providerId: "ollama",
				modelId: "llama3.1:8b",
				reasoning: { enabled: false },
			},
			context: { contextWindow: 65536 },
			portable: "none",
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 65536 } },
					lacks: ["think", "reasoningEffort", "reasoning"],
				},
				{ bucket: "openaiCompatible", lacks: ["reasoningEffort", "reasoning"] },
			],
		},
		{
			name: "ollama unregistered deepseek-r1 explicit reasoning enabled -> portable medium",
			request: {
				providerId: "ollama",
				modelId: "deepseek-r1:latest",
				reasoning: { enabled: true },
			},
			portable: "medium",
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 32768 } },
					lacks: ["think", "reasoningEffort", "reasoning"],
				},
				{ bucket: "openaiCompatible", lacks: ["think", "reasoning"] },
			],
		},
		{
			name: "ollama unregistered model with unset reasoning omits think",
			request: {
				providerId: "ollama",
				modelId: "local-unknown:latest",
			},
			expect: [
				{
					bucket: "ollama",
					has: { options: { num_ctx: 32768 } },
					lacks: ["think", "reasoningEffort", "reasoning"],
				},
				{ bucket: "openaiCompatible", lacks: ["think", "reasoning"] },
			],
		},
	]);
});

describe("composeAiSdkProviderOptions: catalog-driven provider codecs", () => {
	it.each([
		[true, "enabled"],
		[false, "disabled"],
	] as const)("maps Moonshot toggle %s to thinking.type=%s", (enabled, type) => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "moonshot",
				modelId: "kimi-k3",
				reasoning: { enabled },
			}),
			makeContext({
				providerId: "moonshot",
				modelId: "kimi-k3",
				family: "kimi-k3",
				reasoningOptions: [{ type: "toggle" }],
			}),
		);
		if (enabled) {
			expect(result.moonshot).not.toHaveProperty("thinking");
			expect(result.openaiCompatible).not.toHaveProperty("thinking");
		} else {
			expect(result.moonshot).toMatchObject({ thinking: { type } });
			expect(result.openaiCompatible).toMatchObject({ thinking: { type } });
		}
		expect(result.moonshot).not.toHaveProperty("effort");
		expect(result.moonshot).not.toHaveProperty("reasoningSummary");
	});

	it.each([
		// Catalog effort/enablement rides portable reasoning (max -> xhigh).
		[
			"effort",
			{ effort: "max" },
			effortOptions(["low", "medium", "high", "max"]),
			"xhigh",
			undefined,
		],
		[
			"on",
			{ enabled: true },
			[{ type: "toggle" }, ...effortOptions(["low", "medium", "high", "max"])],
			"medium",
			undefined,
		],
		// Explicit disable and exact budgets keep their native wire shapes:
		// @ai-sdk/openai-compatible drops the portable "none" on the floor.
		[
			"off",
			{ enabled: false },
			[{ type: "toggle" }],
			undefined,
			{ reasoningEffort: "none" },
		],
		[
			"budget",
			{ budgetTokens: 4096 },
			budgetOptions(128, 32_768),
			undefined,
			{ thinking: { type: "enabled", budget_tokens: 4096 } },
		],
	] as const)("maps Fireworks %s to its supported wire shape", (_, reasoning, reasoningOptions, portable, expected) => {
		const gatewayRequest = makeRequest({
			providerId: "fireworks",
			modelId: "accounts/fireworks/models/kimi-k3",
			reasoning,
		});
		const gatewayContext = makeContext({
			providerId: "fireworks",
			modelId: "accounts/fireworks/models/kimi-k3",
			reasoningOptions,
		});
		const result = composeAiSdkProviderOptions(gatewayRequest, gatewayContext);
		expect(resolvePortableReasoning(gatewayRequest, gatewayContext)).toBe(
			portable,
		);
		if (expected === undefined) {
			expect(result.fireworks).not.toHaveProperty("reasoningEffort");
			expect(result.fireworks).not.toHaveProperty("thinking");
			return;
		}
		expect(result.fireworks).toMatchObject(expected);
		expect(result.fireworks).not.toHaveProperty("effort");
		expect(result.fireworks).not.toHaveProperty("reasoningSummary");
		if ("thinking" in expected) {
			expect(result.fireworks).not.toHaveProperty("reasoningEffort");
		} else {
			expect(result.fireworks).not.toHaveProperty("thinking");
		}
	});

	it("maps Together off to reasoning.enabled=false", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "together",
				modelId: "zai-org/glm-5.2",
				reasoning: { enabled: false },
			}),
			makeContext({
				providerId: "together",
				modelId: "zai-org/glm-5.2",
				family: "glm",
				reasoningOptions: [{ type: "toggle" }],
			}),
		);
		expect(result.together).toMatchObject({
			reasoning: { enabled: false },
		});
		expect(result.together).not.toHaveProperty("thinking");
	});

	it("keeps Vercel Gemini budget metadata out of Anthropic headroom", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "vercel-ai-gateway",
				modelId: "google/gemini-2.5-pro",
				maxTokens: 64,
				reasoning: { budgetTokens: 128 },
			}),
			makeContext({
				providerId: "vercel-ai-gateway",
				modelId: "google/gemini-2.5-pro",
				family: "gemini-pro",
				reasoningOptions: budgetOptions(128, 32_768),
				maxOutputTokens: 64,
			}),
		);
		for (const bucket of ["vercel-ai-gateway", "vercelAiGateway"]) {
			expect(result[bucket]).toMatchObject({
				reasoning: { max_tokens: 128 },
			});
			expect(result[bucket]).not.toHaveProperty("thinking");
		}
	});
});

describe("composeAiSdkProviderOptions: provider-specific overlays", () => {
	it.each([
		"openai",
		"openai-native",
	])("emits truncation for native OpenAI provider %s", (providerId) => {
		const result = composeAiSdkProviderOptions(
			makeRequest({ providerId, modelId: "gpt-5.4" }),
			makeContext({ providerId, modelId: "gpt-5.4" }),
		);

		expect(result.openai).toHaveProperty("truncation", "auto");
	});

	it("keeps portable OpenAI reasoning out of provider options", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "openai-native",
				modelId: "gpt-5.6",
				reasoning: { effort: "max" },
			}),
			makeContext({
				providerId: "openai-native",
				modelId: "gpt-5.6",
				reasoningOptions: effortOptions([
					"none",
					"low",
					"medium",
					"high",
					"xhigh",
					"max",
				]),
			}),
		);

		expect(result.openai).toEqual(
			expect.objectContaining({
				truncation: "auto",
			}),
		);
		expect(result.openai).not.toHaveProperty("reasoningEffort");
		expect(result).not.toHaveProperty("openai-native");
	});

	it("keeps portable OpenAI disable out of provider options", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "openai-native",
				modelId: "gpt-5.6",
				reasoning: { enabled: false },
			}),
			makeContext({
				providerId: "openai-native",
				modelId: "gpt-5.6",
				reasoningOptions: effortOptions([
					"none",
					"low",
					"medium",
					"high",
					"xhigh",
					"max",
				]),
			}),
		);

		expect(result.openai).toEqual(
			expect.objectContaining({
				truncation: "auto",
			}),
		);
		expect(result.openai).not.toHaveProperty("reasoningEffort");
	});

	it("emits the openai-codex `openai` bucket alongside provider-id and alias buckets", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "openai-codex",
				modelId: "gpt-5.4",
				systemPrompt: "you are helpful",
				reasoning: { effort: "high" },
			}),
			makeContext({
				providerId: "openai-codex",
				modelId: "gpt-5.4",
				reasoningOptions: effortOptions(["low", "medium", "high", "xhigh"]),
			}),
		);

		expect(result.openai).toEqual(
			expect.objectContaining({
				instructions: "you are helpful",
				store: false,
				systemMessageMode: "remove",
			}),
		);
		expect(result.openai).not.toHaveProperty("truncation");
		expect(result["openai-codex"]).toEqual(
			expect.objectContaining({
				store: false,
			}),
		);
		expect(result["openai-codex"]).not.toHaveProperty("reasoningEffort");
		expect(result["openai-codex"]).not.toHaveProperty("reasoningSummary");
		expect(result["openai-codex"]).not.toHaveProperty("truncation");
		expect(result.openaiCodex).toEqual(
			expect.objectContaining({ store: false }),
		);
		expect(result.openaiCodex).not.toHaveProperty("truncation");
	});

	it("keeps portable OpenAI Codex effort out of every provider bucket", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "openai-codex",
				modelId: "gpt-5.4",
				reasoning: { effort: "max" },
			}),
			makeContext({
				providerId: "openai-codex",
				modelId: "gpt-5.4",
				reasoningOptions: effortOptions(["low", "medium", "high", "xhigh"]),
			}),
		);

		for (const bucket of ["openai", "openai-codex", "openaiCodex"]) {
			expect(result[bucket]).not.toHaveProperty("effort");
			expect(result[bucket]).not.toHaveProperty("reasoningEffort");
		}
	});

	it("derives catalog budgets for budget-mode Gemini effort requests", () => {
		const withEffort = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "gemini",
				modelId: "gemini-2.5-flash",
				reasoning: { effort: "medium" },
			}),
			makeContext({
				providerId: "gemini",
				modelId: "gemini-2.5-flash",
				reasoningOptions: [
					{ type: "toggle" },
					{ type: "budget_tokens", min: 0, max: 24_576 },
				],
			}),
		);
		// The catalog advertises only budget controls, so the effort request
		// resolves to a catalog-scaled exact budget with visible thoughts.
		expect(withEffort.google).toEqual({
			thinkingConfig: { thinkingBudget: 12_288, includeThoughts: true },
		});

		const withoutEffort = composeAiSdkProviderOptions(
			makeRequest({ providerId: "gemini", modelId: "gemini-2.5-flash" }),
			makeContext({ providerId: "gemini", modelId: "gemini-2.5-flash" }),
		);
		expect(withoutEffort).not.toHaveProperty("google");
	});

	it("keeps exact Gemini token budgets in provider options", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "gemini",
				modelId: "gemini-2.5-flash",
				reasoning: { budgetTokens: 4096 },
			}),
			makeContext({
				providerId: "gemini",
				modelId: "gemini-2.5-flash",
				reasoningOptions: budgetOptions(0, 24_576),
			}),
		);

		expect(result.google).toEqual({
			thinkingConfig: { thinkingBudget: 4096, includeThoughts: true },
		});
	});

	it("complements portable Gemini levels with visible thoughts only", () => {
		const withMinimal = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "gemini",
				modelId: "gemini-3-pro-preview",
				reasoning: { effort: "minimal" },
			}),
			makeContext({
				providerId: "gemini",
				modelId: "gemini-3-pro-preview",
				reasoningOptions: effortOptions(["low", "high"]),
			}),
		);
		// The level itself rides the portable option; provider options only
		// request thought visibility, which the AI SDK merges on top.
		expect(withMinimal.google).toEqual({
			thinkingConfig: { includeThoughts: true },
		});
	});

	it("derives a catalog budget for budget-mode Google effort requests", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "google",
				modelId: "gemini-2.5-flash",
				reasoning: { enabled: true, effort: "high" },
			}),
			makeContext({
				providerId: "google",
				modelId: "gemini-2.5-flash",
				reasoningOptions: [
					{ type: "toggle" },
					{ type: "budget_tokens", min: 0, max: 24_576 },
				],
			}),
		);

		expect(result.google).toEqual({
			thinkingConfig: { thinkingBudget: 19_660, includeThoughts: true },
		});
	});

	it("complements portable Vertex Gemini effort with visible thoughts", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "vertex",
				modelId: "gemini-3-flash-preview",
				reasoning: { enabled: true, effort: "high" },
			}),
			makeContext({
				providerId: "vertex",
				modelId: "gemini-3-flash-preview",
				reasoningOptions: effortOptions(["minimal", "low", "medium", "high"]),
			}),
		);

		expect(result.vertex).toEqual({
			thinkingConfig: { includeThoughts: true },
		});
		expect(result.vertex).not.toHaveProperty("thinking");
		expect(result.vertex).not.toHaveProperty("effort");
		expect(result.vertex).not.toHaveProperty("reasoningEffort");
		expect(result.vertex).not.toHaveProperty("reasoningSummary");
		expect(result.google).toBeUndefined();
	});

	it("uses portable reasoning for Vertex Claude", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "vertex",
				modelId: "claude-sonnet-4-5",
				reasoning: { enabled: true, effort: "high" },
			}),
			makeContext({
				providerId: "vertex",
				modelId: "claude-sonnet-4-5",
				family: "claude-sonnet",
				capabilities: ["text", "reasoning"],
			}),
		);

		expect(result.vertex).toEqual({});
		expect(result.vertex).not.toHaveProperty("thinkingConfig");
		expect(result.vertex).not.toHaveProperty("effort");
		expect(result.vertex).not.toHaveProperty("reasoningEffort");
		expect(result.vertex).not.toHaveProperty("reasoningSummary");
	});

	it("omits disabled Vertex thinking when the model advertises no off control", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "vertex",
				modelId: "gemini-3-flash-preview",
				reasoning: { enabled: false },
			}),
			makeContext({
				providerId: "vertex",
				modelId: "gemini-3-flash-preview",
				reasoningOptions: effortOptions(["minimal", "low", "medium", "high"]),
			}),
		);

		expect(result.vertex).toEqual({});
		expect(result.vertex).not.toHaveProperty("thinking");
		expect(result.vertex).not.toHaveProperty("effort");
		expect(result.vertex).not.toHaveProperty("reasoningEffort");
		expect(result.vertex).not.toHaveProperty("reasoningSummary");
		expect(result.google).toBeUndefined();
	});

	it("uses portable reasoning to disable Vertex Gemini Flash", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "vertex",
				modelId: "gemini-flash-latest",
				reasoning: { enabled: false },
			}),
			makeContext({
				providerId: "vertex",
				modelId: "gemini-flash-latest",
				family: "gemini-flash",
				capabilities: ["reasoning"],
				reasoningOptions: [
					{ type: "toggle" },
					{ type: "budget_tokens", min: 0, max: 24_576 },
				],
			}),
		);

		expect(result.vertex).toEqual({});
		expect(result.vertex).not.toHaveProperty("thinking");
		expect(result.vertex).not.toHaveProperty("effort");
		expect(result.vertex).not.toHaveProperty("reasoningEffort");
		expect(result.vertex).not.toHaveProperty("reasoningSummary");
		expect(result.google).toBeUndefined();
	});

	it("uses portable reasoning for Vertex Gemini Flash effort", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "vertex",
				modelId: "gemini-flash-latest",
				reasoning: { enabled: true, effort: "high" },
			}),
			makeContext({
				providerId: "vertex",
				modelId: "gemini-flash-latest",
				family: "gemini-flash",
				capabilities: ["reasoning"],
				reasoningOptions: effortOptions(["minimal", "low", "medium", "high"]),
			}),
		);

		expect(result.vertex).toEqual({
			thinkingConfig: { includeThoughts: true },
		});
		expect(result.vertex).not.toHaveProperty("thinking");
		expect(result.vertex).not.toHaveProperty("effort");
		expect(result.vertex).not.toHaveProperty("reasoningEffort");
		expect(result.vertex).not.toHaveProperty("reasoningSummary");
		expect(result.google).toBeUndefined();
	});

	it("omits disabled Vertex Gemini 2.5 Pro without an advertised off control", () => {
		const result = composeAiSdkProviderOptions(
			makeRequest({
				providerId: "vertex",
				modelId: "gemini-2.5-pro",
				reasoning: { enabled: false },
			}),
			makeContext({
				providerId: "vertex",
				modelId: "gemini-2.5-pro",
				family: "gemini-pro",
				capabilities: ["reasoning"],
				reasoningOptions: budgetOptions(128, 32_768),
			}),
		);

		expect(result.vertex).toEqual({});
		expect(result.vertex).not.toHaveProperty("thinking");
		expect(result.vertex).not.toHaveProperty("effort");
		expect(result.vertex).not.toHaveProperty("reasoningEffort");
		expect(result.vertex).not.toHaveProperty("reasoningSummary");
		expect(result.google).toBeUndefined();
	});
});
