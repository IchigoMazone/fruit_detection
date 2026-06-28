import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "Fruit Detection",
  description: "Detect fruit in images, videos, and live camera frames."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
