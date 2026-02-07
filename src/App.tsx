import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import ComputeCPI from "./pages/ComputeCPI";
import Methodology from "./pages/Methodology";
import Memos from "./pages/Memos";
import MemoDetail from "./pages/MemoDetail";
import Work from "./pages/Work";
import About from "./pages/About";
import Contact from "./pages/Contact";
import MarketIntel from "./pages/MarketIntel";
import Calculator from "./pages/Calculator";
import GovBenchmarks from "./pages/GovBenchmarks";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/compute-cpi" element={<ComputeCPI />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/memos" element={<Memos />} />
          <Route path="/memos/:slug" element={<MemoDetail />} />
          <Route path="/work" element={<Work />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/market-intel" element={<MarketIntel />} />
          <Route path="/calculator" element={<Calculator />} />
          <Route path="/gov" element={<GovBenchmarks />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
