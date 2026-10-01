const LOCAL_BACKEND = "http://127.0.0.1:8000";

export function backendBase(): URL {
  const configured = process.env.KYNOVAR_API_INTERNAL?.trim() || LOCAL_BACKEND;
  const url = new URL(configured);
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error("KYNOVAR_API_INTERNAL must use http or https");
  }
  url.pathname = `${url.pathname.replace(/\/$/, "")}/`;
  return url;
}

export function backendUrl(path: string[], search = ""): URL {
  const target = backendBase();
  const suffix = path.map(encodeURIComponent).join("/");
  target.pathname = `${target.pathname}${suffix}`;
  target.search = search;
  return target;
}

export function websocketBase(requestUrl: string): string {
  const explicit = process.env.KYNOVAR_WS_PUBLIC?.trim();
  if (explicit) return explicit.replace(/\/$/, "");

  const backend = backendBase();
  const directlyReachable = backend.protocol === "https:" || ["localhost", "127.0.0.1", "::1"].includes(backend.hostname);
  if (directlyReachable) {
    backend.protocol = backend.protocol === "https:" ? "wss:" : "ws:";
    backend.pathname = `${backend.pathname.replace(/\/$/, "")}/ws`;
    return backend.toString().replace(/\/$/, "");
  }

  const sameOrigin = new URL(requestUrl);
  sameOrigin.protocol = sameOrigin.protocol === "https:" ? "wss:" : "ws:";
  sameOrigin.pathname = "/api/ws";
  sameOrigin.search = "";
  return sameOrigin.toString().replace(/\/$/, "");
}
