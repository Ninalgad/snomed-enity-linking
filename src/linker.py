from gensim.models.keyedvectors import KeyedVectors
from tqdm.notebook import tqdm


def clean_entity(t):
    t = t.lower()
    t = t.replace(' \n', " ")
    t = t.replace('\n', " ")
    return t


class Linker:
    def __init__(self, encoder, context_window_width=0):
        self.encoder = encoder
        self.entity_index = KeyedVectors(self.encoder[1].word_embedding_dimension)
        self.context_index = dict()
        self.history = dict()
        self.context_window_width = context_window_width

    def add_context(self, row):
        window_start = max(0, row.start - self.context_window_width)
        window_end = min(row.end + self.context_window_width, len(row.text))
        return row.text[window_start:window_end]

    def add_entity(self, row):
        return clean_entity(row.text[row.start: row.end])

    def fit(self, df=None, snomed_concepts=None):
        # Create a map from the entities to the concepts and contexts in which they appear
        if df is not None:
            for row in df.itertuples():
                entity = self.add_entity(row)
                context = self.add_context(row)
                map_ = self.history.get(entity, dict())
                contexts = map_.get(row.concept_id, list())
                contexts.append(context)
                map_[row.concept_id] = contexts
                self.history[entity] = map_

        # Add SNOMED CT codes for lookup
        if snomed_concepts is not None:
            for c in snomed_concepts:
                for syn in c.synonyms:
                    map_ = self.history.get(syn, dict())
                    contexts = map_.get(c.sctid, list())
                    contexts.append(syn)
                    map_[c.sctid] = contexts
                    self.history[syn] = map_

        # Create indexes to help disambiguate entities by their contexts
        for entity, map_ in tqdm(self.history.items()):
            keys = [
                (concept_id, occurance)
                for concept_id, contexts in map_.items()
                for occurance, context in enumerate(contexts)
            ]
            contexts = [context for contexts in map_.values() for context in contexts]
            vectors = self.encoder.encode(contexts)
            index = KeyedVectors(self.encoder[1].word_embedding_dimension)
            index.add_vectors(keys, vectors)
            self.context_index[entity] = index

        # Now create the top-level entity index
        keys = list(self.history.keys())
        vectors = self.encoder.encode(keys)
        self.entity_index.add_vectors(keys, vectors)

    def link(self, row):
        entity = self.add_entity(row)
        context = self.add_context(row)
        vec = self.encoder.encode(entity)
        nearest_entity = self.entity_index.most_similar(vec, topn=1)[0][0]
        index = self.context_index.get(nearest_entity, None)

        if index:
            vec = self.encoder.encode(context)
            key, score = index.most_similar(vec, topn=1)[0]
            sctid, _ = key
            return sctid
        else:
            return None
