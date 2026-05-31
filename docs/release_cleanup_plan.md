# HiAntitoxinPred GitHub Release Cleanup Plan

This workspace has completed the first migration pass. The copied files still
contain some historical internal names and absolute-path assumptions from the
development workspace.

## Next Cleanup Tasks

1. Rename historical model symbols and docs-facing terms from the internal
   development names to HiAntitoxinPred terminology.
2. Replace imports of `legacy development workspace.*` with `hiantitoxinpred.*`.
3. Replace absolute development paths with repository-relative paths.
4. Reduce the training runner to the Full Model public workflow.
5. Finalize prediction, embedding extraction, training, and evaluation CLIs.
6. Add smoke tests for import, dataset loading, metrics, and toy prediction.
7. Verify that no large generated artifacts are committed.

