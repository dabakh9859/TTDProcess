import { copyFile, readFile, writeFile } from 'node:fs/promises';

// Apply the shared visual layer without rebuilding the incomplete React views.
const root = new URL('../', import.meta.url);
const indexPath = new URL('dist/index.html', root);
let html = await readFile(indexPath, 'utf8');
const link = '<link rel="stylesheet" href="/professional.css" />';
if (!html.includes('href="/professional.css"')) {
  if (!html.includes('</head>')) throw new Error('dist/index.html: missing </head>');
  html = html.replace('</head>', `  ${link}\n  </head>`);
}
await copyFile(new URL('public/professional.css', root), new URL('dist/professional.css', root));
await writeFile(indexPath, html);
console.log('Interface mise à jour dans dist/ ; bundle applicatif conservé.');
