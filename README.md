# NGC CLI + BTCV Swin-UNETR Setup on RunPod

## 1. Context / Assumptions

- **Environment:** RunPod GPU instance  
- **Project root on pod:** `/workspace/ccmr`  
- **Goal:** Use NGC CLI and BTCV Swin-UNETR pretrained weights (`monai_swin_unetr_btcv_segmentation_0.5.6`).

---

## 2. Install NGC CLI inside the RunPod container

From **inside the pod**:

```bash
cd /workspace/ccmr

# Clean any previous broken attempts
rm -rf ngc-cli ngccli ngccli_linux.zip
rm -f /usr/local/bin/ngc

# Download NGC CLI zip from NVIDIA
wget https://ngc.nvidia.com/downloads/ngccli_linux.zip -O ngccli_linux.zip

# Unzip into a dedicated folder
unzip ngccli_linux.zip -d ngc-cli
Now you should have:

ls ngc-cli
# expected: ngc, libpython3.11.so.1.0, other support files
Important: Do not move ngc out of this folder. It needs the local .so libraries next to it.

Add it to PATH for the current shell:

export PATH=/workspace/ccmr/ngc-cli:$PATH
Test:

ngc --help
If you see the help text, NGC CLI is installed correctly.

3. Configure NGC CLI with your API key
In a browser, go to NGC and sign in.

Go to Setup → API Key, generate an API key, and copy it.

Back in the RunPod shell:

export NGC_CLI_API_KEY="<YOUR_NGC_API_KEY_HERE>"
ngc config set
Accept defaults for most prompts; just ensure the API key is set.

Note: For monai_swin_unetr_btcv_segmentation, the CLI download-version call returned 403 Access Denied. So the direct CLI download is not usable for this specific model with the current NGC org permissions. The CLI is still useful in general, just not for this particular checkpoint.

4. Download BTCV Swin-UNETR bundle manually from NGC
Since

ngc registry model download-version nvidia/monai_swin_unetr_btcv_segmentation:0.5.6
returned 403, the model was downloaded manually via the web UI.

On the local Windows machine:

Go to the model page in a browser:
monai_swin_unetr_btcv_segmentation (team monaitoolkit, version 0.5.6).

Click Download in the NGC UI.

Save the file, e.g.:
C:\Users\aryan\Downloads\monai_swin_unetr_btcv_segmentation_0.5.6.zip

Extract it locally (right-click → Extract All…).
You get a folder:
C:\Users\aryan\Downloads\monai_swin_unetr_btcv_segmentation_0.5.6\

Inside that folder is the MONAI bundle structure (configs, docs, models, etc.).

5. Transfer the extracted bundle to RunPod via scp
5.1 Get SSH details from RunPod
From the RunPod UI, under SSH access, you see a command like:

bash
Copy code
ssh root@194.68.245.209 -p 22106 -i ~/.ssh/id_ed25519
This gives you:

Host: 194.68.245.209

Port: 22106

Key path (on local machine): C:\Users\aryan\.ssh\id_ed25519

5.2 Use scp from Windows PowerShell
On the Windows machine (not inside the container):

powershell
Copy code
scp -P 22106 `
  -i "C:\Users\aryan\.ssh\id_ed25519" `
  -r "C:\Users\aryan\Downloads\monai_swin_unetr_btcv_segmentation_0.5.6" `
  root@194.68.245.209:/workspace/ccmr/pretrained/
After this, on the pod you have:

text
Copy code
/workspace/ccmr/pretrained/monai_swin_unetr_btcv_segmentation_0.5.6
6. Locate the actual model.pt inside the bundle
The NVIDIA bundle is nested a bit. On the pod:

bash
Copy code
cd /workspace/ccmr/pretrained
ls
# monai_swin_unetr_btcv_segmentation_0.5.6

cd monai_swin_unetr_btcv_segmentation_0.5.6
ls
# monai_swin_unetr_btcv_segmentation_0.5.6

cd monai_swin_unetr_btcv_segmentation_0.5.6
ls
# _

cd _
ls
# LICENSE  configs  docs  models

cd models
ls
# model.pt
So the final checkpoint path is:

text
Copy code
pretrained/monai_swin_unetr_btcv_segmentation_0.5.6/monai_swin_unetr_btcv_segmentation_0.5.6/_/models/model.pt
This is the file used for Swin-UNETR fine-tuning.

7. Example: use model.pt for Swin-UNETR fine-tuning
From /workspace/ccmr with your venv active:

bash
Copy code
source .venv/bin/activate
Run the L=5 finetune smoke test:

bash
Copy code
python -u scripts/train_swin_unetr_finetune_btcv.py \
  --data_root data/processed/hvsmr2 \
  --train_split data/splits/train_L5.txt \
  --val_split data/splits/val_ids.txt \
  --roi_size 96 96 96 \
  --batch_size 1 \
  --epochs 10 \
  --lr 1e-4 \
  --num_workers 2 \
  --pretrained_ckpt pretrained/monai_swin_unetr_btcv_segmentation_0.5.6/monai_swin_unetr_btcv_segmentation_0.5.6/_/models/model.pt \
  --output_dir logs/swin_unetr/L5_finetune_smoketest
During startup, you should see something like:

Dropping mismatched keys (e.g., head): ['out.conv.conv.bias', 'out.conv.conv.weight']
Loaded BTCV Swin-UNETR weights for fine-tuning. Missing: ['out.conv.conv.weight', 'out.conv.conv.bias'], Unexpected: []

This is expected: it loads the BTCV encoder/decoder, skips the old head, and uses the 9-class head.

8. Quick notes / gotchas to remember
If you see ModuleNotFoundError: No module named 'monai', you probably ran python outside .venv. Reactivate with:

bash
Copy code
source .venv/bin/activate
Do not move ngc out of the ngc-cli folder. Instead, add that folder to PATH.

For the BTCV Swin-UNETR model, ngc registry model download-version ... returns 403, so manual browser download + scp is the working method.

Disk path to checkpoint in this exact setup:

text
Copy code
/workspace/ccmr/pretrained/monai_swin_unetr_btcv_segmentation_0.5.6/monai_swin_unetr_btcv_segmentation_0.5.6/_/models/model.pt
perl
Copy code
