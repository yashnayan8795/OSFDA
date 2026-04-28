import React from "react"
import type {Metadata} from 'next'
import {Geist, Geist_Mono} from 'next/font/google'
import {Analytics} from '@vercel/analytics/next'
import './globals.css'
import {ApolloProvider} from '@/lib/apollo/provider'

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans"
});
const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono"
});

export const metadata: Metadata = {
    title: 'OSFDA Analytics — Aviation Safety Intelligence',
    description: 'Integrated ML system for aviation incident severity, categorization, pre-flight risk, emerging risks, and factor analysis.',
    generator: 'v0.app',
    icons: {
        icon: [
            {
                url: '/icon.svg',
                type: 'image/svg+xml',
            },
        ],
        apple: '/apple-icon.png',
    },
}

export default function RootLayout({
                                       children,
                                   }: Readonly<{
    children: React.ReactNode
}>) {
    return (
        <html lang="en" className={`dark ${geist.variable} ${geistMono.variable}`}>
        <body className="bg-background text-foreground font-sans antialiased">
        <ApolloProvider>
          {children}
        </ApolloProvider>
        <Analytics/>
        </body>
        </html>
    )
}
