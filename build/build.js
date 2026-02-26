const { buildPythonTarget } = require("./pipeline/pythonTarget");
const { buildWebTarget } = require("./pipeline/webTarget");

function parseOption(argv, key) {
  const keyEquals = `--${key}=`;
  const keyFlag = `--${key}`;
  const direct = argv.find((arg) => arg.startsWith(keyEquals));
  if (direct) {
    return direct.slice(keyEquals.length);
  }
  const idx = argv.indexOf(keyFlag);
  if (idx >= 0 && idx + 1 < argv.length) {
    return argv[idx + 1];
  }
  return null;
}

function parsePositionals(argv, knownValueOptions) {
  const positionals = [];
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      const optionName = arg.includes("=") ? arg.slice(2, arg.indexOf("=")) : arg.slice(2);
      if (!arg.includes("=") && knownValueOptions.has(optionName)) {
        i += 1;
      }
      continue;
    }
    positionals.push(arg);
  }
  return positionals;
}

function printResult(title, result, extraLabel, extraPath) {
  console.log(`${title} complete.`);
  console.log("Common artifacts:");
  console.log(`- ${result.commonArtifacts.appBundlePath}`);
  console.log(`- ${result.commonArtifacts.inlineHtmlPath}`);
  console.log(`${extraLabel}:`);
  console.log(`- ${extraPath}`);
}

async function main() {
  const argv = process.argv.slice(2);
  const target = parseOption(argv, "target");
  const knownValueOptions = new Set(["target", "dest"]);
  const positionals = parsePositionals(argv, knownValueOptions);
  const destinationDir = parseOption(argv, "dest") || (target === "web" ? positionals[0] ?? null : null);

  if (target === "python") {
    const result = await buildPythonTarget();
    printResult("Python build", result, "Python standalone", result.standaloneScriptPath);
    return;
  }

  if (target === "web") {
    const result = await buildWebTarget({ destinationDir });
    printResult("Web build", result, "Web target placeholder", result.webHtmlPath);
    console.log("React wrapper:");
    console.log(`- ${result.reactComponentPath}`);
    if (result.destinationSync) {
      console.log("Synced destination:");
      console.log(`- ${result.destinationSync.resolvedDest}`);
      console.log("Copied files:");
      for (const file of result.destinationSync.copied) {
        console.log(`- ${file}`);
      }
    }
    return;
  }

  console.error("Invalid or missing target.");
  console.error("Usage: node build/build.js --target python");
  console.error("   or: node build/build.js --target web [--dest <folder>]");
  console.error("   or: npm run build:web -- <folder>");
  process.exit(1);
}

main().catch((error) => {
  console.error("build/build.js failed");
  console.error(error);
  process.exit(1);
});
