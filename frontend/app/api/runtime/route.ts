import { websocketBase } from "@/lib/server/backend";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export function GET(request: Request): Response {
  return Response.json({ websocket_base: websocketBase(request.url) });
}
