import Navbar from './components/layout/Navbar'
import HeroSection from './components/hero/HeroSection'
import HowItWorks from './components/how-it-works/HowItWorks'
import ModelRegistry from './components/model-registry/ModelRegistry'
import Workspace from './components/workspace/Workspace'
import Footer from './components/layout/Footer'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col bg-white text-navy selection:bg-saffron-100 selection:text-saffron-900">
      <Navbar />
      <HeroSection />
      <HowItWorks />
      <ModelRegistry />
      <Workspace />
      <Footer />
    </div>
  )
}
