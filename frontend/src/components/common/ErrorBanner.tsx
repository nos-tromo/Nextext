export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-foreground">
      {message}
    </div>
  )
}
