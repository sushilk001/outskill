# 🔧 Chrome Browser Troubleshooting Guide

## ✅ Current Status
- **Streamlit UI:** Running on http://localhost:8501
- **Access Page:** `open_ui.html` should be open in your browser

---

## 🌐 How to Access the UI

### Method 1: Click the Streamlit Button
On the page that just opened, click the "Open Streamlit UI" button

### Method 2: Manual URL
1. Open Chrome
2. Type in address bar: `http://localhost:8501`
3. Press Enter

---

## 🔴 Common Chrome Issues & Fixes

### Issue 1: "This site can't be reached"
**Fixes:**
```bash
# Check if Streamlit is running
curl http://localhost:8501

# If not running, start it:
streamlit run app.py
```

### Issue 2: Chrome shows blank page
**Fixes:**
1. **Hard refresh:** Press `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
2. **Clear cache:**
   - Press `Ctrl+Shift+Delete` (Windows) or `Cmd+Shift+Delete` (Mac)
   - Select "Cached images and files"
   - Click "Clear data"
3. **Try incognito:** Press `Ctrl+Shift+N` (Windows) or `Cmd+Shift+N` (Mac)

### Issue 3: Connection refused or ERR_CONNECTION_REFUSED
**Fixes:**
1. Verify Streamlit is running:
```bash
ps aux | grep streamlit
```

2. If not running, restart:
```bash
cd /Users/sushil/sushil-workspace/outskill
streamlit run app.py
```

3. Wait 5-10 seconds, then refresh Chrome

### Issue 4: SSL/HTTPS errors
**Fix:** Use `http://` NOT `https://`
- ✅ Correct: `http://localhost:8501`
- ❌ Wrong: `https://localhost:8501`

### Issue 5: Extensions blocking
**Fixes:**
1. Disable ad blockers temporarily
2. Disable privacy extensions (Privacy Badger, uBlock Origin, etc.)
3. Try incognito mode (extensions are usually disabled there)

### Issue 6: Port already in use
**Fix:** Use a different port
```bash
# Stop existing Streamlit
pkill -f streamlit

# Start on different port
streamlit run app.py --server.port 8502
```
Then access: `http://localhost:8502`

---

## 🎯 Alternative Solutions

### Try Gradio Instead (Often More Reliable)
```bash
# In terminal:
python app_gradio.py

# Then open in Chrome:
http://localhost:7860
```

### Try Different Browser
- **Safari:** Usually works well on Mac
- **Firefox:** Good alternative
- **Edge:** Available on Mac now

---

## 🐛 Debug Commands

### Check if port is accessible:
```bash
curl -v http://localhost:8501
```

### Check what's running on port 8501:
```bash
lsof -i :8501
```

### View Streamlit logs:
```bash
tail -f ~/Library/Logs/streamlit/*.log
```

### Restart everything:
```bash
# Kill Streamlit
pkill -f streamlit

# Wait a moment
sleep 2

# Restart
cd /Users/sushil/sushil-workspace/outskill
streamlit run app.py
```

---

## 📋 Step-by-Step Chrome Access

1. **Open Chrome browser**

2. **Clear any cached data** (Optional but recommended):
   - Press `Cmd+Shift+Delete`
   - Check "Cached images and files"
   - Time range: "Last hour"
   - Click "Clear data"

3. **Type the URL**:
   ```
   http://localhost:8501
   ```

4. **Press Enter**

5. **Wait 5-10 seconds** for the app to load

6. **You should see**:
   - Title: "🤖 RAG Persona Search System"
   - Chat input at the bottom
   - Sidebar on the left

---

## ✨ What to Do When It Works

1. Try a sample query from the sidebar
2. Type your own question in the chat
3. Adjust settings (search mode, number of results)
4. Explore different search modes

---

## 🆘 Still Not Working?

### Quick Alternative - Command Line Interface
```bash
# Use the terminal interface instead:
python test_rag.py

# Or quick test:
python quick_test.py "Who are the AI experts?"
```

### Check System Requirements
```bash
# Verify installations:
python --version  # Should be 3.8+
streamlit --version
ollama --version

# Check if database exists:
ls -la lancedb_data/
```

---

## 📞 Last Resort Solutions

### 1. Fresh Start
```bash
# Kill everything
pkill -f streamlit
pkill -f ollama

# Remove cache
rm -rf ~/.streamlit/

# Restart Ollama
ollama serve &

# Restart Streamlit
cd /Users/sushil/sushil-workspace/outskill
streamlit run app.py
```

### 2. Use Gradio Instead
Gradio is often more reliable:
```bash
python app_gradio.py
```
Access at: http://localhost:7860

### 3. Use Terminal Interface
Most reliable option:
```bash
python test_rag.py
```

---

## 💡 Pro Tips

1. **Bookmark the URL:** Add `http://localhost:8501` to bookmarks
2. **Keep terminal open:** Don't close the terminal running Streamlit
3. **First load is slow:** Model initialization takes 10-15 seconds
4. **Incognito works best:** Fewer issues with cache and extensions
5. **Check firewall:** Some firewalls block localhost connections

---

## ✅ Success Checklist

- [ ] Streamlit is running (`ps aux | grep streamlit` shows process)
- [ ] Port 8501 is accessible (`curl http://localhost:8501` works)
- [ ] Chrome is updated to latest version
- [ ] Using `http://` not `https://`
- [ ] Tried hard refresh (`Cmd+Shift+R`)
- [ ] Tried incognito mode
- [ ] No firewall blocking localhost

---

If all else fails, the command-line interface (`python test_rag.py`) and Gradio (`python app_gradio.py`) are excellent alternatives that avoid browser issues entirely!



