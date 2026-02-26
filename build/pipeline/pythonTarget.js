const fs = require("node:fs/promises");
const path = require("node:path");
const {
  DIST_DIR,
  buildCommonArtifacts,
  ensureDir,
  renderPythonStandaloneScript,
} = require("./common");

const DIST_PYTHON_DIR = path.join(DIST_DIR, "python");

async function buildPythonTarget() {
  const commonArtifacts = await buildCommonArtifacts();
  await ensureDir(DIST_PYTHON_DIR);

  const standaloneScriptPath = path.join(DIST_PYTHON_DIR, "standalone_server.py");
  const standaloneScript = renderPythonStandaloneScript(commonArtifacts.inlineHtml);
  await fs.writeFile(standaloneScriptPath, standaloneScript, "utf8");

  return {
    distPythonDir: DIST_PYTHON_DIR,
    standaloneScriptPath,
    commonArtifacts,
  };
}

module.exports = {
  buildPythonTarget,
};
