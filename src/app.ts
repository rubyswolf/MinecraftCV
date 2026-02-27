type McvOperation = "cv.opencvTest";

type McvRequest<TArgs = Record<string, unknown>> = {
  op: McvOperation;
  args: TArgs;
};

type McvSuccess<TData> = {
  ok: true;
  data: TData;
};

type McvFailure = {
  ok: false;
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
};

type McvResponse<TData> = McvSuccess<TData> | McvFailure;

type McvOpencvTestResult = {
  opencv_version: string;
  gray_values: number[];
  shape: number[];
  mean_gray: number;
};

type McvMediaApi = {
  url: string;
  available: () => boolean;
  fetch: () => Promise<Response>;
};

type McvDataApi = {
  url: string;
  available: () => boolean;
};

type McvClientApi = {
  media: McvMediaApi;
  data: McvDataApi;
  mcv: {
    call: typeof callMcvApi;
    backend: "python" | "web";
  };
};

type MediaTab = "videos" | "images";
type MediaLoadState = "loading" | "no_api" | "fetching" | "failed" | "loaded";

type MediaVideoEntry = {
  name: string;
  url: string;
  youtube_id?: string;
};

type MediaImageEntry = {
  name: string;
  url: string;
};

type MediaLibrary = {
  videos: Record<string, MediaVideoEntry>;
  images: Record<string, MediaImageEntry>;
};

type SelectedMedia = {
  tab: MediaTab;
  id: string;
  item: MediaVideoEntry | MediaImageEntry;
};

declare global {
  interface Window {
    MCV_API?: McvClientApi;
  }
}

declare const __MCV_BACKEND__: "python" | "web";
declare const __MCV_OPENCV_URL__: string;
declare const __MCV_MEDIA_API_URL__: string;
declare const __MCV_DATA_API_URL__: string;

let cvPromise: Promise<unknown> | null = null;
let activeMediaTab: MediaTab = "videos";
let mediaLoadState: MediaLoadState = "loading";
let mediaSearchQuery = "";
let selectedMedia: SelectedMedia | null = null;
let mediaLibrary: MediaLibrary = {
  videos: {},
  images: {},
};

function isThenable(value: unknown): value is Promise<unknown> {
  return typeof value === "object" && value !== null && "then" in value;
}

function loadScriptOnce(scriptUrl: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-mcv-src="${scriptUrl}"]`) as
      | HTMLScriptElement
      | null;
    if (existing) {
      if (existing.dataset.loaded === "1") {
        resolve();
        return;
      }
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error(`Failed to load ${scriptUrl}`)), {
        once: true,
      });
      return;
    }

    const script = document.createElement("script");
    script.src = scriptUrl;
    script.async = true;
    script.dataset.mcvSrc = scriptUrl;
    script.addEventListener(
      "load",
      () => {
        script.dataset.loaded = "1";
        resolve();
      },
      { once: true }
    );
    script.addEventListener("error", () => reject(new Error(`Failed to load ${scriptUrl}`)), {
      once: true,
    });
    document.head.appendChild(script);
  });
}

async function getWebMcvRuntime(): Promise<any> {
  if (!cvPromise) {
    cvPromise = (async () => {
      const maybeGlobalCv = (globalThis as any).cv;
      if (!maybeGlobalCv) {
        await loadScriptOnce(__MCV_OPENCV_URL__);
      }
      const cvCandidate = (globalThis as any).cv;
      if (!cvCandidate) {
        throw new Error("opencv.js runtime not found on window.cv");
      }
      if (typeof cvCandidate === "function") {
        return await cvCandidate();
      }
      if (isThenable(cvCandidate)) {
        return await cvCandidate;
      }
      return cvCandidate;
    })();
  }
  return cvPromise;
}

async function callMcvApi<TData>(requestBody: McvRequest): Promise<McvResponse<TData>> {
  if (__MCV_BACKEND__ === "web") {
    if (requestBody.op !== "cv.opencvTest") {
      return {
        ok: false,
        error: {
          code: "UNKNOWN_OP",
          message: `Unsupported operation in web backend: ${requestBody.op}`,
        },
      };
    }

    try {
      const cv = await getWebMcvRuntime();
      const src = cv.matFromArray(1, 3, cv.CV_8UC3, [255, 0, 0, 0, 255, 0, 0, 0, 255]);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
      const grayValues = Array.from(gray.data as Uint8Array).map((value) => Number(value));
      const response = {
        ok: true,
        data: {
          opencv_version: String((cv as any).VERSION ?? "opencv.js"),
          gray_values: grayValues,
          shape: [Number(gray.rows), Number(gray.cols)],
          mean_gray:
            grayValues.length > 0
              ? grayValues.reduce((sum, value) => sum + value, 0) / grayValues.length
              : 0,
        },
      } satisfies McvSuccess<McvOpencvTestResult>;
      src.delete();
      gray.delete();
      return response as McvResponse<TData>;
    } catch (error) {
      return {
        ok: false,
        error: {
          code: "WEB_BACKEND_ERROR",
          message: "OpenCV.js call failed",
          details: String(error),
        },
      };
    }
  }

  try {
    const response = await fetch("/api/mcv", {
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

    return (await response.json()) as McvResponse<TData>;
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

async function fetchMediaApi(): Promise<Response> {
  return fetch(__MCV_MEDIA_API_URL__);
}

function isMediaApiAvailable(): boolean {
  return typeof __MCV_MEDIA_API_URL__ === "string" && __MCV_MEDIA_API_URL__.trim().length > 0;
}

function isDataApiAvailable(): boolean {
  return typeof __MCV_DATA_API_URL__ === "string" && __MCV_DATA_API_URL__.trim().length > 0;
}

function installGlobalApi(): void {
  window.MCV_API = {
    media: {
      url: __MCV_MEDIA_API_URL__,
      available: isMediaApiAvailable,
      fetch: fetchMediaApi,
    },
    data: {
      url: __MCV_DATA_API_URL__,
      available: isDataApiAvailable,
    },
    mcv: {
      call: callMcvApi,
      backend: __MCV_BACKEND__,
    },
  };
}

function getMediaBoxNode(): HTMLDivElement | null {
  return document.getElementById("media-box") as HTMLDivElement | null;
}

function setMediaMessage(message: string): void {
  const mediaBoxNode = getMediaBoxNode();
  if (!mediaBoxNode) {
    return;
  }
  mediaBoxNode.textContent = message;
}

function configureExternalLink(node: HTMLAnchorElement, url: string, label: string): void {
  node.className = "media-link";
  node.href = url;
  node.target = "_blank";
  node.rel = "noopener noreferrer";
  node.textContent = label;
  node.addEventListener("click", (event) => {
    event.preventDefault();
    window.open(url, "_blank", "noopener,noreferrer");
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeVideos(value: unknown): Record<string, MediaVideoEntry> {
  if (!isRecord(value)) {
    return {};
  }
  const output: Record<string, MediaVideoEntry> = {};
  for (const [id, entry] of Object.entries(value)) {
    if (!isRecord(entry)) {
      continue;
    }
    const name = typeof entry.name === "string" ? entry.name.trim() : "";
    const url = typeof entry.url === "string" ? entry.url.trim() : "";
    if (!name || !url) {
      continue;
    }
    const youtubeId =
      typeof entry.youtube_id === "string" && entry.youtube_id.trim()
        ? entry.youtube_id.trim()
        : undefined;
    output[id] = {
      name,
      url,
      ...(youtubeId ? { youtube_id: youtubeId } : {}),
    };
  }
  return output;
}

function normalizeImages(value: unknown): Record<string, MediaImageEntry> {
  if (!isRecord(value)) {
    return {};
  }
  const output: Record<string, MediaImageEntry> = {};
  for (const [id, entry] of Object.entries(value)) {
    if (!isRecord(entry)) {
      continue;
    }
    const name = typeof entry.name === "string" ? entry.name.trim() : "";
    const url = typeof entry.url === "string" ? entry.url.trim() : "";
    if (!name || !url) {
      continue;
    }
    output[id] = { name, url };
  }
  return output;
}

function parseMediaLibrary(payload: unknown): MediaLibrary | null {
  if (!isRecord(payload)) {
    return null;
  }
  return {
    videos: normalizeVideos(payload.videos),
    images: normalizeImages(payload.images),
  };
}

function setTabButtonState(): void {
  const videosButton = document.getElementById("tab-videos") as HTMLButtonElement | null;
  const imagesButton = document.getElementById("tab-images") as HTMLButtonElement | null;
  if (!videosButton || !imagesButton) {
    return;
  }
  const isVideos = activeMediaTab === "videos";
  videosButton.classList.toggle("active", isVideos);
  imagesButton.classList.toggle("active", !isVideos);
}

function normalizeSearchToken(value: string): string {
  return value.trim().toLowerCase();
}

function matchesSearch(haystackValues: string[]): boolean {
  const query = normalizeSearchToken(mediaSearchQuery);
  if (!query) {
    return true;
  }
  return haystackValues.some((value) => normalizeSearchToken(value).includes(query));
}

function renderMediaList(): void {
  const mediaBoxNode = getMediaBoxNode();
  if (!mediaBoxNode) {
    return;
  }
  const rawMediaEntries =
    activeMediaTab === "videos"
      ? Object.entries(mediaLibrary.videos)
      : Object.entries(mediaLibrary.images);
  const mediaEntries =
    activeMediaTab === "videos"
      ? rawMediaEntries.filter(([id, item]) =>
          matchesSearch([
            id,
            item.name,
            item.url,
            "youtube_id" in item && item.youtube_id ? item.youtube_id : "",
          ])
        )
      : rawMediaEntries.filter(([id, item]) => matchesSearch([id, item.name, item.url]));

  if (mediaEntries.length === 0) {
    const hasQuery = normalizeSearchToken(mediaSearchQuery).length > 0;
    if (hasQuery) {
      setMediaMessage(activeMediaTab === "videos" ? "No matching videos found." : "No matching images found.");
      return;
    }
    setMediaMessage(activeMediaTab === "videos" ? "No videos found." : "No images found.");
    return;
  }

  const listNode = document.createElement("ul");
  listNode.className = "media-list";

  for (const [id, item] of mediaEntries) {
    const listItemNode = document.createElement("li");
    listItemNode.className = "media-item";

    const titleNode = document.createElement("button");
    titleNode.type = "button";
    titleNode.className = "media-title-button";
    titleNode.textContent = item.name;
    titleNode.addEventListener("click", () => {
      selectedMedia = { tab: activeMediaTab, id, item };
      renderMediaBox();
    });

    const metaNode = document.createElement("div");
    metaNode.className = "media-meta";
    metaNode.textContent = id;
    if (activeMediaTab === "videos" && "youtube_id" in item && item.youtube_id) {
      const separatorNode = document.createTextNode(" | YouTube: ");
      const youtubeLinkNode = document.createElement("a");
      const youtubeUrl = `https://www.youtube.com/watch?v=${encodeURIComponent(item.youtube_id)}`;
      configureExternalLink(youtubeLinkNode, youtubeUrl, item.youtube_id);
      metaNode.appendChild(separatorNode);
      metaNode.appendChild(youtubeLinkNode);
    }

    const urlNode = document.createElement("a");
    configureExternalLink(urlNode, item.url, item.url);

    listItemNode.appendChild(titleNode);
    listItemNode.appendChild(metaNode);
    listItemNode.appendChild(urlNode);
    listNode.appendChild(listItemNode);
  }

  mediaBoxNode.replaceChildren(listNode);
}

function renderSelectedMediaPlaceholder(): void {
  const mediaBoxNode = getMediaBoxNode();
  if (!mediaBoxNode || !selectedMedia) {
    return;
  }

  const container = document.createElement("div");
  container.className = "selected-media";

  const titleNode = document.createElement("div");
  titleNode.className = "selected-title";
  titleNode.textContent = `Selected ${selectedMedia.tab === "videos" ? "Video" : "Image"}: ${selectedMedia.item.name}`;

  const idNode = document.createElement("div");
  idNode.className = "selected-line";
  idNode.textContent = `ID: ${selectedMedia.id}`;

  const urlNode = document.createElement("a");
  configureExternalLink(urlNode, selectedMedia.item.url, selectedMedia.item.url);

  const backButton = document.createElement("button");
  backButton.type = "button";
  backButton.className = "back-button";
  backButton.textContent = "Back";
  backButton.addEventListener("click", () => {
    selectedMedia = null;
    renderMediaBox();
  });

  container.appendChild(titleNode);
  container.appendChild(idNode);
  container.appendChild(urlNode);
  container.appendChild(backButton);
  mediaBoxNode.replaceChildren(container);
}

function renderMediaBox(): void {
  if (mediaLoadState === "no_api") {
    setMediaMessage("No Media API");
    return;
  }
  if (mediaLoadState === "fetching") {
    setMediaMessage("Fetching...");
    return;
  }
  if (mediaLoadState === "failed") {
    setMediaMessage("Failed to fetch Media Library");
    return;
  }
  if (mediaLoadState === "loaded") {
    if (selectedMedia) {
      renderSelectedMediaPlaceholder();
      return;
    }
    renderMediaList();
    return;
  }
  setMediaMessage("loading...");
}

function setActiveMediaTab(nextTab: MediaTab): void {
  activeMediaTab = nextTab;
  selectedMedia = null;
  setTabButtonState();
  renderMediaBox();
}

function installUiHandlers(): void {
  const videosButton = document.getElementById("tab-videos") as HTMLButtonElement | null;
  const imagesButton = document.getElementById("tab-images") as HTMLButtonElement | null;
  const searchInput = document.getElementById("media-search") as HTMLInputElement | null;
  if (videosButton) {
    videosButton.addEventListener("click", () => {
      setActiveMediaTab("videos");
    });
  }
  if (imagesButton) {
    imagesButton.addEventListener("click", () => {
      setActiveMediaTab("images");
    });
  }
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      mediaSearchQuery = searchInput.value;
      renderMediaBox();
    });
  }
}

async function loadMediaLibrary(): Promise<void> {
  if (!isMediaApiAvailable()) {
    mediaLoadState = "no_api";
    renderMediaBox();
    return;
  }

  mediaLoadState = "fetching";
  renderMediaBox();

  try {
    const response = await fetchMediaApi();
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const parsed = parseMediaLibrary(await response.json());
    if (!parsed) {
      throw new Error("Invalid media schema");
    }
    mediaLibrary = parsed;
    mediaLoadState = "loaded";
    renderMediaBox();
  } catch {
    mediaLoadState = "failed";
    renderMediaBox();
  }
}

function bootstrap(): void {
  installGlobalApi();
  installUiHandlers();
  setTabButtonState();
  renderMediaBox();
  void loadMediaLibrary();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}
