# Tomodachi upstream rendering review — 2026-10-29

User observation: Mii shading currently looks normal; the control/cursor still
projects several rectangular patches onto grass. Green flickering persists.
The new screenshot is not evidence that every Mii shadow is fixed.

Reviewed the parent repository's public open and closed PR listing and searched
issues for shadow, flicker, green, rectangle and Tomodachi. No Tomodachi-specific
report or exact green-horizontal-flicker fix was found. Similar artifacts occur
in other ports, but that does not establish a single bug affecting every game.

## Relevant merged fixes and local comparison

| Upstream change | Evidence and limits | Local status / applicability |
| --- | --- | --- |
| [#67](https://github.com/ly05010419/MacNativeGames/pull/67), [#155](https://github.com/ly05010419/MacNativeGames/pull/155) | TOTK ground decals read stale attachment contents; initial pass splitting was followed by tile reads for overlapping decals. In-game before/after evidence. #155 also has a Metal 3 synthetic GPU test. | Local renderer lacks `att_unstored_read` / `att_fb_take`. Existing explicit barrier handling does not establish correctness for games that omit barriers between attachment-sampling draws. Strong investigation candidate for stale UI shapes, not a proven Tomodachi diagnosis. |
| [#547](https://github.com/ly05010419/MacNativeGames/pull/547) | Argument-cache invalidation now includes texture serial replacement and retired uniform snapshots. Synthetic production-path GPU regression; no gameplay claim. | Local legacy `args_sync_dirty` checks only registry and dirty generations. The newer optional cache key already includes serial/snapshot generations, but that does not fix the legacy path. Concrete shared-renderer correctness gap. |
| [#104](https://github.com/ly05010419/MacNativeGames/pull/104) | D16 shadow compare-reference rounding and nearest non-compare depth reads fixed TOTK dark patches/cascade blocks. Tested TOTK 1.0.0. | Local `totk_msl_fixup.h` absent. Do not apply its assumption that array comparisons are D16 to another game without checking the actual texture format. |
| [#222](https://github.com/ly05010419/MacNativeGames/pull/222) | Preserves the NVN fragment control section's early-depth-test flag; fixed dotted lattice surfaces in TOTK. | Local translator/runtime do not forward `early_fragment_tests`. Candidate for depth artifacts; not proof about pure-green pixels or cursor shadows. Requires shader-control layout validation and cache migration. |
| [#193](https://github.com/ly05010419/MacNativeGames/pull/193) | Larger byte snapshots distinguish writes to neighboring data from writes to terrain normal maps; fixes skipped write-backs. | Relevant resource-alias/write-back mechanism, but current grass texture recovery came from the separate full bindless-table fix. Avoid treating the old flat terrain and current cursor patches as identical. |
| [#842](https://github.com/ly05010419/MacNativeGames/pull/842) | Smash gained shader-emulated fixed-function alpha testing. | Local code still reports alpha testing as unimplemented. No matching alpha-test warning was found in the inspected normal Tomodachi profile log; not established as the cursor cause. |

These are software correctness fixes used by the Metal 4 parent, not evidence
that switching API versions automatically solves the symptoms. In particular,
#155's standalone test explicitly runs on Metal 3. The user's Metal 3 PR
[#6](https://github.com/ly05010419/MacNativeGames/pull/6) is closed and unmerged;
the local branch does not automatically inherit later parent fixes.

## Follow-up order

1. Port and exercise #547's cache invalidation regression on Metal 3.
2. Trace the cursor's attachment reads and resource identities, then adapt the
   attachment-read ordering/tile-read fixes with their GPU tests. Preserve full
   8192-entry bindless handling and the verified Mii mip fixes.
3. Verify depth formats and shader flags before applying TOTK-specific shader
   transformations. Investigate green pixels separately unless a capture ties
   them to the same draw/resource failure.

No runtime or packaged app was modified by this review; none of the candidate
changes has yet been validated against the remaining Tomodachi artifacts.
Public PR metadata and diffs were read into `/tmp/mng-prs-*.json` and
`/tmp/mng-pr-*-files.json`, outside the source tree.

## Cache invalidation backport and Metal 3 result

Ported #547's production fix and its synthetic regression into the shared
renderer. Before the fix, the regression aborted at mode 0, snapshot 0,
stripe 8: red channel 96 instead of 191 (the replaced texture stayed cached).
After tracking both serial and snapshot-retirement generations, the complete
`probe-gpu-arg-epoch` passed 23,354 checks with `MTL_DEBUG_LAYER=1` and
`TOTK_RENDER_STREAM=0`. The new regression covers five cache configurations.
The packaged Desktop app was replaced after archiving the preceding app and
passed strict codesign verification.

Visual check: green horizontal specks remain visible in the yellow loading
transition with this fix. A room's Mii face still renders normally. This is a
confirmed correctness fix, not a claim that the reported graphics symptoms
are resolved. Build/test logs: `/tmp/tomodachi-cache-{before,after}.log` and
`/tmp/tomodachi-cache-build-fixed.log`. Session evidence is outside the source
under `Development/Tomodachi_build/host/tests/upstream-cache-fix-qj3mireg`.
