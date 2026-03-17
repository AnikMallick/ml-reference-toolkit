import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home        from "./components/HomePage";
import Sklearn_app from "./components/SklearnTable";
import Feature_app from "./components/FeatureEngTable";
import Prob_app    from "./components/ProblemsTable";
import EDA_app     from "./components/EdaTable";
import Play_app    from "./components/Playbook";
import Deploy_app  from "./components/Deploy";
import Torch_app   from "./components/PyTorch";
 
export default function App() {
  return (
    <BrowserRouter basename="/ml-reference-toolkit">
      <Routes>
        <Route path="/"         element={<Home />}        />
        <Route path="/sklearn"  element={<Sklearn_app />} />
        <Route path="/features" element={<Feature_app />} />
        <Route path="/problems" element={<Prob_app />}    />
        <Route path="/eda"      element={<EDA_app />}     />
        <Route path="/playbook" element={<Play_app />}    />
        <Route path="/deploy"   element={<Deploy_app />}  />
        <Route path="/torch"    element={<Torch_app />}   />
      </Routes>
    </BrowserRouter>
  );
}