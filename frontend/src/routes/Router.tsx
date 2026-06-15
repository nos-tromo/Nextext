import { Route, Routes } from 'react-router-dom'
import { Shell } from '../components/layout/Shell'
import { Home } from './Home'

export function Router() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
    </Shell>
  )
}
