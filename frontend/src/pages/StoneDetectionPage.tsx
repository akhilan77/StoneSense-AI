import { ImageUpload } from '../components/ImageUpload';

export function StoneDetectionPage() {
  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-800 bg-slate-900/50 p-5">
        <h1 className="text-2xl font-semibold text-white">Stone Detection</h1>
        <p className="mt-2 text-sm text-slate-400">Upload a supported medical image to receive the existing mock detection response.</p>
      </section>
      <ImageUpload />
    </div>
  );
}
