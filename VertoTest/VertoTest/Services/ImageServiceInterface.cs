using VertoTest.Models;

namespace VertoTest.Services
{
    public interface ImageServiceInterface
    {
        Task<bool> Exists(int imageId);
        Task<bool> Add(ImageViewModel content);
        Task<Image?> GetViaId(int? id);
        Task<Image?> GetViaFilename(string fileName);
        Task<bool> Edit(Image image);
        Task<bool> Delete(Image image);
        Task<List<Image>> List();
    }
}
