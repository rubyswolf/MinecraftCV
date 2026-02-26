type CvOperation = "opencvTest";

type CvRequest<TArgs = Record<string, unknown>> = {
  op: CvOperation;
  args: TArgs;
};

type CvSuccess<TData> = {
  ok: true;
  data: TData;
};

type CvFailure = {
  ok: false;
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
};

type CvResponse<TData> = CvSuccess<TData> | CvFailure;

type OpencvTestResult = {
  opencv_version: string;
  gray_values: number[];
  shape: number[];
  mean_gray: number;
};

async function callCvApi<TData>(requestBody: CvRequest): Promise<CvResponse<TData>> {
  try {
    const response = await fetch("/api/cv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      return {
        ok: false,
        error: {
          code: "HTTP_ERROR",
          message: `HTTP ${response.status}`,
        },
      };
    }

    return (await response.json()) as CvResponse<TData>;
  } catch (error) {
    return {
      ok: false,
      error: {
        code: "NETWORK_ERROR",
        message: "Could not reach backend API",
        details: String(error),
      },
    };
  }
}

function setStatus(message: string): void {
  const statusNode = document.getElementById("status");
  if (statusNode) {
    statusNode.textContent = message;
  }
}

function setOutput(message: string): void {
  const outputNode = document.getElementById("output");
  if (outputNode) {
    outputNode.textContent = message;
  }
}

async function checkHealth(): Promise<void> {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      setStatus(`Backend health check failed: HTTP ${response.status}`);
      return;
    }
    const payload = (await response.json()) as { ok: boolean; backend?: string };
    if (!payload.ok) {
      setStatus("Backend health check failed");
      return;
    }
    setStatus(`Connected to backend (${payload.backend ?? "unknown"})`);
  } catch {
    setStatus("Could not connect to backend");
  }
}

async function handleTestButtonClick(): Promise<void> {
  const button = document.getElementById("test-button") as HTMLButtonElement | null;
  if (!button) {
    return;
  }

  button.disabled = true;
  setStatus("Running OpenCV test...");

  const result = await callCvApi<OpencvTestResult>({
    op: "opencvTest",
    args: {},
  });

  if (result.ok) {
    setStatus(`OpenCV call succeeded (version ${result.data.opencv_version})`);
    setOutput(JSON.stringify(result.data, null, 2));
  } else {
    setStatus(`OpenCV call failed: ${result.error.code}`);
    setOutput(JSON.stringify(result.error, null, 2));
  }

  button.disabled = false;
}

function installUiHandlers(): void {
  const button = document.getElementById("test-button");
  if (button) {
    button.addEventListener("click", () => {
      void handleTestButtonClick();
    });
  }
}

function bootstrap(): void {
  installUiHandlers();
  void checkHealth();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}
