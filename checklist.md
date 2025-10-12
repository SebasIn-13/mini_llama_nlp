
## Assignment Checklist

### Setup & Environment
- [ ] Use `bash setup.sh` as guidance for installation, but feel free to change the versions as needed
- [ ] Verify model weights downloaded (stories42M.pt)

### Core Implementation
- [ ] **llama.py**
- [ ] **rope.py**
- [ ] **optimizer.py**
- [ ] **classifier.py**
- [ ] **lora.py**

### Testing & Validation
- [ ] Pass `python sanity_check.py`
- [ ] Pass `python optimizer_test.py` 
- [ ] Pass `python rope_test.py` 
- [ ] Generate coherent text with `python run_llama.py --option generate`
- [ ] Complete SST zero-shot prompting
- [ ] Complete CFIMDB zero-shot prompting  
- [ ] Complete SST fine-tuning
- [ ] Complete CFIMDB fine-tuning
- [ ] Complete SST LoRA fine-tuning
- [ ] Complete CFIMDB LoRA fine-tuning

