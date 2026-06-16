export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-muted-foreground" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-border border-t-primary" />
      {label && <span>{label}</span>}
    </div>
  )
}
