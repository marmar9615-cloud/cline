import type {
	GatewayProviderContext,
	GatewayStreamRequest,
} from "@cline/shared";
import type { CallSettings } from "ai";
import { getModelReasoningControls } from "../model-facts";
import { normalizeReasoningRequest } from "./reasoning-options";

export type AiSdkReasoning = NonNullable<CallSettings["reasoning"]>;

/**
 * Providers whose AI SDK package maps the full portable scale, including the
 * explicit-disable value "none". Providers routed through
 * `@ai-sdk/openai-compatible` (deepseek, fireworks, groq, xai, ...) are
 * deliberately absent: that package forwards effort levels as
 * `reasoning_effort` but silently drops "none", so an explicit disable would
 * vanish from the wire. Their disable requests must keep riding the native
 * provider-option rules (e.g. DeepSeek `thinking.type = "disabled"`).
 */
const PORTABLE_REASONING_DISABLE_PROVIDERS = new Set([
	"anthropic",
	"bedrock",
	"gemini",
	"google",
	"ollama",
	"openai-codex",
	"openai-native",
	"vertex",
]);

const NON_PORTABLE_REASONING_PROVIDERS = new Set([
	"claude-code",
	"dify",
	"mistral",
	"opencode",
	"sapaicore",
]);

/**
 * Resolve reasoning intent owned by the AI SDK's portable top-level option.
 *
 * Effort and enablement intent is normalized against the catalog-advertised
 * model controls first: the AI SDK's own portable mapping is catalog-unaware,
 * so without this step out-of-catalog values (e.g. `xhigh` on a model that
 * advertises `low|high|max`) would reach the wire, and models whose catalog
 * advertises no user-facing control would receive spurious reasoning settings.
 * Explicit disable skips normalization on purpose so it always wins, even for
 * models with no advertised "off" control.
 */
export function resolvePortableReasoning(
	request: GatewayStreamRequest,
	context: GatewayProviderContext,
): AiSdkReasoning | undefined {
	const reasoning = request.reasoning;
	if (!reasoning) {
		return undefined;
	}
	// Models that declare capabilities without "reasoning" must not receive
	// reasoning settings at all (custom models leave capabilities undefined
	// and are not affected).
	const capabilities = context.model.capabilities;
	if (capabilities !== undefined && !capabilities.includes("reasoning")) {
		return undefined;
	}
	if (reasoning.enabled === false) {
		return PORTABLE_REASONING_DISABLE_PROVIDERS.has(request.providerId)
			? "none"
			: undefined;
	}
	// Exact token budgets always stay with provider-specific options, even
	// when normalization would downgrade them (e.g. to a toggle enable) for
	// models without an advertised budget control.
	if (typeof reasoning.budgetTokens === "number") {
		return undefined;
	}
	const normalized = normalizeReasoningRequest(request, context).reasoning;
	if (!normalized) {
		return undefined;
	}
	if (typeof normalized.budgetTokens === "number") {
		return undefined;
	}
	if (NON_PORTABLE_REASONING_PROVIDERS.has(request.providerId)) {
		return undefined;
	}
	if (normalized.effort) {
		return normalized.effort === "max" ? "xhigh" : normalized.effort;
	}
	// Models advertising a models.dev "default" effort take their provider's
	// default instead of an invented portable level; the native rules encode
	// it (e.g. reasoning_effort "default" on Groq-hosted Qwen).
	if (
		getModelReasoningControls(context.model.reasoningOptions)?.supportsDefault
	) {
		return undefined;
	}
	if (normalized.enabled !== true) {
		return undefined;
	}
	// Toggle-only models advertise no levels; preserve the requested effort
	// intent with the broadly supported clamp used for unlisted models.
	if (reasoning.effort) {
		return reasoning.effort === "minimal"
			? "low"
			: reasoning.effort === "xhigh" || reasoning.effort === "max"
				? "high"
				: reasoning.effort;
	}
	return "medium";
}

/**
 * Remove portable intent before provider options are composed. AI SDK ignores
 * top-level reasoning whenever reasoning controls also occur in providerOptions.
 */
export function withoutPortableReasoning(
	request: GatewayStreamRequest,
	context: GatewayProviderContext,
): GatewayStreamRequest {
	const normalizedRequest =
		request.reasoning?.enabled === false &&
		(request.reasoning.effort !== undefined ||
			request.reasoning.budgetTokens !== undefined)
			? { ...request, reasoning: { enabled: false } }
			: request;
	return resolvePortableReasoning(normalizedRequest, context)
		? { ...normalizedRequest, reasoning: undefined }
		: normalizedRequest;
}
