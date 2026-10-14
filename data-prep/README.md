# Data prep

From the repo root:

```bash
make label                                  # label viewer GUI on data/sms_labeled.jsonl
make label DATA=data/other.jsonl            # ...or on another file
make augment N=200                          # synthetic spam via a local Ollama model
```

This environment uses the system Python (`python-preference = "only-system"`):
uv's bundled Python has a Tk build that crashes on Fedora/Wayland.
On Fedora, tkinter comes from the `python3-tkinter` package.
