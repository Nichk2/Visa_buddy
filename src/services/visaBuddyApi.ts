import type { ChatRequest, ChatResponse, ApiHealth } from '../types/visa_buddy';

const API_BASE_URL = import.meta.env.VITE_VISA_BUDDY_API_URL || '';

export class VisaBuddyApiService {
    private baseUrl: string;

    constructor(baseUrl?: string) {
        this.baseUrl = baseUrl || API_BASE_URL;
    }

    async checkHealth(): Promise<ApiHealth> {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            throw new Error(`Health check failed: ${error}`);
        }
    }

    async sendMessage(message: string, maxLength: number = 700): Promise<string> {
        try {
            const request: ChatRequest = {
                message,
                max_length: maxLength
            };

            const response = await fetch(`${this.baseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data: ChatResponse = await response.json();
            return data.response;
        } catch (error) {
            throw new Error(`Failed to send message: ${error}`);
        }
    }

    setBaseUrl(url: string): void {
        this.baseUrl = url;
    }

    getBaseUrl(): string {
        return this.baseUrl;
    }
}

export const visaBuddyApi = new VisaBuddyApiService();