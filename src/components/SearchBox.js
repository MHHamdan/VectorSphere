import { useState } from "react";
import axios from "axios";

const SearchBox = ({ onResults }) => {
  const [query, setQuery] = useState("");

  const handleSearch = async (event) => {
    event.preventDefault();
    if (!query) return;

    try {
      const response = await axios.post("http://localhost:8000/api/hybrid-search/", {
        query,
        k: 5,
      });
      onResults(response.data.results);
    } catch (error) {
      console.error("Error searching:", error);
    }
  };

  return (
    <div className="flex flex-col items-center mt-5">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter search query..."
        className="w-1/2 p-2 border rounded"
      />
      <button onClick={handleSearch} className="mt-3 p-2 bg-blue-500 text-white rounded">
        Search
      </button>
    </div>
  );
};

export default SearchBox;

