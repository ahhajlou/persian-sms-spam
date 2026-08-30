# Spam Filter

## Dump sms from android device

```bash
adb shell content query --uri content://sms/ >sms_dump.txt
```