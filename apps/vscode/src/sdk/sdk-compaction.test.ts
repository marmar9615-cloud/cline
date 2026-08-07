import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"

const createContextCompactionPrepareTurn = vi.fn()
const createSessionCompactionState = vi.fn((input: unknown) => ({ version: 1, input }))
const projectSessionCompactionState = vi.fn()
vi.mock("@cline/core", () => ({
	createContextCompactionPrepareTurn: (...args: unknown[]) => createContextCompactionPrepareTurn(...args),
	createSessionCompactionState: (input: unknown) => createSessionCompactionState(input),
	projectSessionCompactionState: (...args: unknown[]) => projectSessionCompactionState(...args),
}))

vi.mock("@/shared/services/Logger", () => ({
	Logger: { debug: vi.fn(), error: vi.fn(), log: vi.fn(), warn: vi.fn() },
}))

let compactSessionMessages: typeof import("./sdk-compaction").compactSessionMessages

const baseConfig = {
	providerConfig: { providerId: "anthropic", modelId: "claude" },
	providerId: "anthropic",
	modelId: "claude",
	knownModels: { claude: { id: "claude", maxInputTokens: 200_000 } },
	compaction: undefined,
	logger: undefined,
	telemetry: undefined,
} as unknown as Parameters<typeof compactSessionMessages>[0]["config"]

describe("compactSessionMessages", () => {
	beforeAll(async () => {
		;({ compactSessionMessages } = await import("./sdk-compaction"))
	})

	beforeEach(() => {
		vi.clearAllMocks()
	})

	it("returns compacted=false without invoking the SDK when there are no messages", async () => {
		const result = await compactSessionMessages({ config: baseConfig, sessionId: "s1", messages: [] })

		expect(result).toEqual({ compacted: false, messages: [] })
		expect(createContextCompactionPrepareTurn).not.toHaveBeenCalled()
	})

	it("builds a manual-mode prepareTurn and force-enables compaction", async () => {
		const compact = vi
			.fn()
			.mockResolvedValue({ messages: [{ role: "user", content: "summary" }], systemPrompt: "rewritten system" })
		createContextCompactionPrepareTurn.mockReturnValueOnce(compact)

		const messages = [
			{ role: "user" as const, content: "1" },
			{ role: "assistant" as const, content: "2" },
		]
		const result = await compactSessionMessages({ config: baseConfig, sessionId: "s1", messages })

		// Manual mode + enabled compaction + telemetry keying.
		expect(createContextCompactionPrepareTurn).toHaveBeenCalledWith(
			expect.objectContaining({
				providerId: "anthropic",
				modelId: "claude",
				compaction: expect.objectContaining({ enabled: true }),
				sessionId: "s1",
			}),
			{ mode: "manual" },
		)
		expect(compact).toHaveBeenCalledOnce()
		expect(createSessionCompactionState).toHaveBeenCalledWith({
			sourceMessages: messages,
			compactedMessages: [{ role: "user", content: "summary" }],
			conversationId: "s1",
			systemPrompt: "rewritten system",
		})
		expect(result).toEqual({
			compacted: true,
			messages: [{ role: "user", content: "summary" }],
			compactionState: { version: 1, input: expect.anything() },
		})
	})

	it("preserves context-only model limits for the shared resolver", async () => {
		const compact = vi.fn().mockResolvedValue({ messages: [{ role: "user", content: "summary" }] })
		createContextCompactionPrepareTurn.mockReturnValueOnce(compact)
		const contextOnlyConfig = {
			...baseConfig,
			knownModels: { claude: { id: "claude", contextWindow: 400_000 } },
		} as unknown as Parameters<typeof compactSessionMessages>[0]["config"]

		await compactSessionMessages({
			config: contextOnlyConfig,
			sessionId: "s-context-only",
			messages: [{ role: "user", content: "long context" }],
		})

		expect(compact).toHaveBeenCalledWith(
			expect.objectContaining({
				model: expect.objectContaining({
					info: { id: "claude", contextWindow: 400_000 },
				}),
			}),
		)
	})

	it("compacts the sidecar projection instead of the canonical transcript", async () => {
		// cline/cline#12996: the canonical history can exceed the model window
		// by millions of tokens; the working context is what must be compacted.
		const compact = vi.fn().mockResolvedValue({ messages: [{ role: "user", content: "new summary" }] })
		createContextCompactionPrepareTurn.mockReturnValueOnce(compact)
		const canonical = [
			{ role: "user" as const, content: "huge prefix" },
			{ role: "assistant" as const, content: "huge reply" },
			{ role: "user" as const, content: "tail" },
		]
		const projected = [
			{ role: "user" as const, content: "prior summary" },
			{ role: "user" as const, content: "tail" },
		]
		const compactionState = { version: 1, messages: [{ role: "user", content: "prior summary" }] }
		projectSessionCompactionState.mockReturnValueOnce(projected)

		const result = await compactSessionMessages({
			config: baseConfig,
			sessionId: "s1",
			messages: canonical,
			compactionState: compactionState as never,
		})

		expect(projectSessionCompactionState).toHaveBeenCalledWith(compactionState, canonical)
		expect(compact).toHaveBeenCalledWith(expect.objectContaining({ messages: projected, apiMessages: projected }))
		// The new sidecar is still keyed to the canonical transcript.
		expect(createSessionCompactionState).toHaveBeenCalledWith(
			expect.objectContaining({
				sourceMessages: canonical,
				compactedMessages: [{ role: "user", content: "new summary" }],
			}),
		)
		expect(result.compacted).toBe(true)
	})

	it("falls back to the canonical transcript when the sidecar no longer projects", async () => {
		const compact = vi.fn().mockResolvedValue({ messages: [{ role: "user", content: "summary" }] })
		createContextCompactionPrepareTurn.mockReturnValueOnce(compact)
		const canonical = [{ role: "user" as const, content: "only message" }]
		projectSessionCompactionState.mockReturnValueOnce(undefined)

		await compactSessionMessages({
			config: baseConfig,
			sessionId: "s1",
			messages: canonical,
			compactionState: { version: 1, messages: [] } as never,
		})

		expect(compact).toHaveBeenCalledWith(expect.objectContaining({ messages: canonical, apiMessages: canonical }))
	})

	it("carries the prior sidecar's system prompt forward when compacting a projection", async () => {
		const compact = vi.fn().mockResolvedValue({ messages: [{ role: "user", content: "summary" }] })
		createContextCompactionPrepareTurn.mockReturnValueOnce(compact)
		const canonical = [{ role: "user" as const, content: "1" }]
		projectSessionCompactionState.mockReturnValueOnce([{ role: "user" as const, content: "projected" }])

		await compactSessionMessages({
			config: baseConfig,
			sessionId: "s1",
			messages: canonical,
			compactionState: { version: 1, messages: [], system_prompt: "carried system prompt" } as never,
		})

		expect(createSessionCompactionState).toHaveBeenCalledWith(
			expect.objectContaining({ systemPrompt: "carried system prompt" }),
		)
	})

	it("returns compacted=false when prepareTurn is unavailable", async () => {
		createContextCompactionPrepareTurn.mockReturnValueOnce(undefined)

		const messages = [{ role: "user" as const, content: "1" }]
		const result = await compactSessionMessages({ config: baseConfig, sessionId: "s1", messages })

		expect(result).toEqual({ compacted: false, messages })
	})

	it("returns compacted=false when the strategy declines (returns undefined)", async () => {
		const compact = vi.fn().mockResolvedValue(undefined)
		createContextCompactionPrepareTurn.mockReturnValueOnce(compact)

		const messages = [{ role: "user" as const, content: "1" }]
		const result = await compactSessionMessages({ config: baseConfig, sessionId: "s1", messages })

		expect(result).toEqual({ compacted: false, messages })
	})
})
