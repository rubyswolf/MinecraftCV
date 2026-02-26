const fs = require("node:fs/promises");
const path = require("node:path");
const { DIST_DIR, buildCommonArtifacts, ensureDir } = require("./common");

const DIST_WEB_DIR = path.join(DIST_DIR, "web");

function escapeForTemplateLiteral(input) {
  return input.replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$\{/g, "\\${");
}

function renderReactWrapperTsx(inlineHtml) {
  const htmlLiteral = escapeForTemplateLiteral(inlineHtml);
  return `"use client";

import React from "react";

const MCV_HTML = \`${htmlLiteral}\`;

export type MCVProps = {
  className?: string;
  style?: React.CSSProperties;
  title?: string;
};

export default function MCV(props: MCVProps) {
  const { className, style, title = "MinecraftCV" } = props;
  return (
    <iframe
      title={title}
      className={className}
      style={{ width: "100%", minHeight: 480, border: 0, ...style }}
      srcDoc={MCV_HTML}
      sandbox="allow-scripts allow-same-origin allow-forms allow-downloads"
    />
  );
}
`;
}

async function copyToDestination(destDir, artifacts) {
  const resolvedDest = path.resolve(destDir);
  await ensureDir(resolvedDest);

  const copied = [];

  const commonTargets = [
    [artifacts.webHtmlPath, path.join(resolvedDest, "index.html")],
    [artifacts.commonArtifacts.appBundlePath, path.join(resolvedDest, "app.bundle.js")],
    [artifacts.reactComponentPath, path.join(resolvedDest, "MCV.tsx")],
  ];

  for (const [src, dst] of commonTargets) {
    await fs.copyFile(src, dst);
    copied.push(dst);
  }

  return { resolvedDest, copied };
}

async function buildWebTarget(options = {}) {
  const commonArtifacts = await buildCommonArtifacts();
  await ensureDir(DIST_WEB_DIR);

  const webHtmlPath = path.join(DIST_WEB_DIR, "index.html");
  await fs.writeFile(webHtmlPath, commonArtifacts.inlineHtml, "utf8");
  const reactComponentPath = path.join(DIST_WEB_DIR, "MCV.tsx");
  await fs.writeFile(
    reactComponentPath,
    renderReactWrapperTsx(commonArtifacts.inlineHtml),
    "utf8"
  );

  let destinationSync = null;
  if (options.destinationDir) {
    destinationSync = await copyToDestination(options.destinationDir, {
      commonArtifacts,
      webHtmlPath,
      reactComponentPath,
    });
  }

  return {
    distWebDir: DIST_WEB_DIR,
    webHtmlPath,
    reactComponentPath,
    commonArtifacts,
    destinationSync,
  };
}

module.exports = {
  buildWebTarget,
};
