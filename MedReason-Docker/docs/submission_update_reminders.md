# Submission document update reminders

The filled Word file (`data/MedReason26_Model_Description_FILLED.docx`) was completed manually.
**Do not regenerate from the script unless you first copy your team/contact edits back into**
`MedReason-Docker/tools/fill_model_description.py`.

## Critical timeline

**Pre-evaluation is not a dry run.** Organizer-side scores on the hidden public split
**count equally** with the hidden private split toward the **final official ranking**.
Therefore the **pre-evaluation Docker package should already include your best method**:
- LoRA fine-tuned checkpoint (not “planned”)
- Final prompts and parsing
- Validated end-to-end on validation split

Final submission (15 July 2026) can refine the same method; update the PDF if anything changes.

## Update before pre-evaluation submit

- [ ] Docker cloud download link
- [ ] Method section: LoRA **completed** (hyperparameters, training time, checkpoint path)
- [ ] Train MCQ baseline score (local)
- [ ] Hardware actually used
- [ ] Checklist: all items checked after real GPU inference + Docker test
- [ ] Export Word → **PDF**
- [ ] Submission phase: **Pre-evaluation** (or both if organizers allow)

## Update before final submission (15 July 2026)

- [ ] Final Docker link (if image changed)
- [ ] Any method changes after pre-eval feedback
- [ ] Final train/validation experiment summary
- [ ] Submission phase: **Final evaluation**
- [ ] Re-export PDF

## Regenerate Word from script (optional)

Only if you synced your edits into `fill_model_description.py`:

```powershell
cd MedReason-Docker
python tools/fill_model_description.py
```

Then re-apply manual team fields if needed.
