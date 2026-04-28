import { Hero } from "@/components/landing/Hero";
import { KpiBand } from "@/components/landing/KpiBand";
import { ProblemsGrid } from "@/components/landing/ProblemsGrid";
import { ArchitectureSection } from "@/components/landing/ArchitectureSection";
import { LandingFooter } from "@/components/landing/Footer";

export const metadata = {
  title: "OSFDA Analytics — Aviation Safety Intelligence",
  description: "Integrated ML system for incident severity, categorization, pre-flight risk, emerging risks, and factor analysis.",
};

export default function Home() {
  return (
    <>
      <Hero />
      <KpiBand />
      <ProblemsGrid />
      <ArchitectureSection />
      <LandingFooter />
    </>
  );
}
