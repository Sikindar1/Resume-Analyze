import {readFile} from "fs/promises"
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter"
import {HuggingFaceInferenceEmbeddings} from "@langchain/community/embeddings/hf"
import { loadEnvFile } from "process"
import { createClient } from "@supabase/supabase-js"
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase"
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"
import { PromptTemplate } from "@langchain/core/prompts"
import { StringOutputParser } from "@langchain/core/output_parsers"

loadEnvFile()

const resumeText = await readFile('./resume.txt', 'utf-8')

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50
})

const docs = await splitter.createDocuments([resumeText])

const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HF_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2"
})

const client = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY)

// SupabaseVectorStore.fromDocuments(docs, embeddings, {
//     client,
//     tableName: 'documents',
//     queryName: 'match_documents'
// })

const vectorStore = new SupabaseVectorStore(embeddings, {
    client,
    tableName: 'documents',
    queryName: 'match_documents'
})

const retriever = vectorStore.asRetriever()

const retrieveDocs = await retriever.invoke('What are my strenghts based on this resume?')
const context = retrieveDocs.map(doc => doc.pageContent).join('\n\n')

const llm = new ChatGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
    model: "gemini-1.5-flash",
    temperature: 0.7
})

const prompt = PromptTemplate.fromTemplate(`
You are a helpful carrer coach. Based on the following resume context, give constructive feedback about the user's strength and what kinds of roles they are suited for.
Resume:
{context}

Feedback:
`)

const chain = prompt.pipe(llm).pipe(new StringOutputParser())
const result = await chain.invoke({context})

console.log(result)