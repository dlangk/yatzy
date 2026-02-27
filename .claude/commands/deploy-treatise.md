Deploy only the treatise to production. Fast path — no Docker rebuild.

Run `just deploy-treatise` which will:
1. Build treatise markdown → HTML
2. Rsync static files to the server (~3 seconds)
3. Run health check
4. Purge Cloudflare cache

Report the result and the URL: https://langkilde.se/yatzy/
