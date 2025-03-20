import { useState } from "react";
import SearchBox from "./components/SearchBox";
import SearchResults from "./components/SearchResults";

function App() {
  const [results, setResults] = useState([]);

  return (
    <div className="p-10 text-center">
      <h1 className="text-3xl font-bold">VectorSphere Search</h1>
      <SearchBox onResults={setResults} />
      <SearchResults results={results} />
    </div>
  );
}

export default App;

