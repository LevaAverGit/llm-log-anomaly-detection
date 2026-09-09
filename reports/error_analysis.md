# Error analysis

**Where rules miss but the LLM catches it (3 examples):**

- **lx-0044** · linux_auth · truth: incident · T1078 — rules stayed silent, the LLM flagged it. Successful SSH login from an IP that had just brute-forced the host: valid-account access obtained through credential guessing.
  - `Jan 10 09:14:00 web-01 sshd[1500]: Accepted password for root from 203.0.113.99 port 55099 ssh2`
- **win-0019** · windows_security · truth: incident · T1110.001 — rules stayed silent, the LLM flagged it. Burst of Windows failed logons (Event 4625) against administrator/admin from one source, preceding a backdoor account.
  - `{"timestamp": "2026-01-10T09:05:00+00:00", "event_code": "4625", "host": "WIN-SERVER01", "username": "administrator", "source_ip": "192.0.2.150", "failure_reas…`
  - `{"timestamp": "2026-01-10T09:05:05+00:00", "event_code": "4625", "host": "WIN-SERVER01", "username": "administrator", "source_ip": "192.0.2.150", "failure_reas…`
- **win-0021** · windows_security · truth: incident · T1059.001 — rules stayed silent, the LLM flagged it. Encoded PowerShell command line (-enc): obfuscated script execution.
  - `{"timestamp": "2026-01-10T09:15:00+00:00", "event_code": "4688", "host": "WIN-SERVER01", "username": "jsmith", "source_ip": null, "process_name": "powershell.e…`

**Where the LLM misfires but rules are right (3 examples):**

- **cld-0001** · cloud_audit · truth: benign — the LLM raised a false alarm, the rules stayed silent. Routine successful console login.
  - `{"action": "console_login", "status": "success", "username": "sre@example.com", "source_ip": "198.51.100.60", "region": "eu-central-1"}`
- **lx-0002** · linux_auth · truth: benign — the LLM raised a false alarm, the rules stayed silent. Routine successful SSH login by a staff account.
  - `web-01 sshd: Accepted password for backup from 203.0.113.6`
- **ng-0002** · nginx_access · truth: benign — the LLM raised a false alarm, the rules stayed silent. Ordinary page/asset request from a normal browser.
  - `10.0.4.9 "GET /blog" 304`
