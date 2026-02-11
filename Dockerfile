FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

---

### **4. Update Model Loading**

**If using Custom Vision:**
- Export model as ONNX or TensorFlow
- Include model file in repo OR load from Azure endpoint

**If using local model:**
- Include model weights in repo
- Make sure path is relative: `./model.pth` not `/Users/steven/model.pth`

---

### **5. Environment Variables (Optional)**

API keys:

Create `.env` file (don't commit to GitHub):
```
AZURE_KEY=your_key_here
