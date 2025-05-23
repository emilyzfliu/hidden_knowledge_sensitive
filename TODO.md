### ACTION ITEMS
- [ ] Evaluate model outputs on all current prompts, excluding yes/no tests, using open-source HF models. Include synthetic sensitive in context.
- [ ] Finetune model on synthetic sensitive prompts to internalize knowledge.
  - [ ] Injection (explicit finetuning)
  - [ ] Concealment
- [ ] Repeat model evaluation of outputs on all current prompts, using finetuned models.
- [ ] Using yes/no dataset, evaluate difference between model output response and internal knowledge (through Contrast-Consistent Search)
  
### DONE ITEMS
- [x] Generate a list of prompts specifically querying elements in the dataset (yes or no)