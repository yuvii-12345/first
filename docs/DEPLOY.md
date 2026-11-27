# Deploying

Live: **https://zeroth-opal.vercel.app**
Vercel project: `zeroth` (`prj_jAG9BMvffkB0uJtLcHhKfygX5HoI`, team `dhanaojh-7668`)

## What is deployed

A single self-contained file. `npm run site` builds `site/index.html`: the whole
app — React, the field tower, the curve, the pairing, the NTT, KZG, PLONK, and
the Web Worker that runs the prover — inlined into one 292 kB HTML file with no
external requests at all.

That file is committed at `site/index.html` and the Vercel build
(`deploy/build.mjs`) fetches it from the pinned commit and publishes it:

```
deploy/package.json   {"scripts": {"build": "node build.mjs"}}
deploy/build.mjs      fetch site/index.html @ <commit> → dist/index.html
project settings      framework vite · build `node build.mjs` · output `dist` · install `echo skip`
```

The pin lives in `deploy/build.mjs` (`REF`), overridable with the `ZEROTH_REF`
environment variable.

## Why not a normal Git integration

The Vercel account has no GitHub *Login Connection*, so Vercel cannot link
`yuvii-12345/first` and run `npm run build` on its own:

```
400  Failed to link yuvii-12345/first. You need to add a Login Connection
     to your GitHub account first.
```

That link is an OAuth grant only the account owner can make: **Vercel → Account
Settings → Login Connections → GitHub**. Once it exists, the project can be
re-linked (or a new one imported) and every push builds from source instead:

```bash
npx vercel link          # pick the repo
npx vercel --prod        # framework: vite, build: npm run build, output: dist
```

Everything needed for that is already in the repo — `package.json`, `vite.config.ts`,
`index.html`, `src/`. The only reason the current deployment fetches a
prebuilt file instead is the missing OAuth grant.

## Re-deploying after a change

```bash
npm run site                                  # rebuild site/index.html
git add site && git commit -m "site: rebuild" && git push
# bump REF in deploy/build.mjs to the new commit, then upload the two files again
```

Uploading means creating a deployment in the `zeroth` project with
`deploy/package.json` and `deploy/build.mjs` as its files and the project
settings above — the Vercel connector (`create_deployment`) does this in one
call.

## Notes

- `site/` is committed on purpose: it is the deployable artifact, and the only
  way to ship without a Git link. `dist/` stays ignored.
- The site is fully static. No functions, no environment variables, no
  analytics beyond Vercel's defaults.
- Everything runs client-side: the prover, the verifier and the SRS live in the
  visitor's tab. No witness ever leaves the browser, because there is nowhere
  for it to go.
