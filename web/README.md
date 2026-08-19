## One-time setup

1. Put password in `web/protect/.password` (gitignored):

   ```
   echo -n 'your-password-here' > web/protect/.password
   ```

2. Build the gated site into `docs/`:

   ```
   node web/build.js
   ```

## Updating the site

After any change, rebuild and commit:

```
node web/build.js
git add docs
git commit -m "..."
```
