import type { Metadata, Viewport } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Ryan Rochman - Systems Engineer & Quantitative Developer',
  description: 'Terminal portfolio showcasing GPU programming, trading systems, and ML infrastructure.',
  keywords: ['systems engineer', 'quantitative developer', 'GPU programming', 'trading systems', 'CUDA', 'machine learning'],
  authors: [{ name: 'Ryan Rochman' }],
  robots: 'index, follow'
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="min-h-screen bg-terminal-bg text-terminal-text font-mono">
        {children}
      </body>
    </html>
  )
}