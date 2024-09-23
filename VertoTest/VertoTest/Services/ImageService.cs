using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using VertoTest.Context;
using VertoTest.Models;

namespace VertoTest.Services
{
    public class ImageService : ImageServiceInterface
    {
        private readonly VertoAppDbContext context;

        public ImageService(VertoAppDbContext context)
        {
            this.context = context;
        }

        // Checks to see if item already exists in table
        public async Task<bool> Exists(int imageId)
        {
            return await this.context.Images.AnyAsync(image => image.ImageId == imageId);
        }

        // Returns the item with a matching primary key
        public async Task<Image?> GetViaId(int? id)
        {
            if (id == null)
            {
                return null;
            }
            return await this.context.Images.FirstOrDefaultAsync(image => image.ImageId == id);
        }
        // Returns the item with a matching file name
        public async Task<Image?> GetViaFilename(string fileName)
        {
            return await this.context.Images.FirstOrDefaultAsync(image => image.FileName == fileName);
        }

        // Handles table edit/updates 
        public async Task<bool> Edit(Image image)
        {
            var item = await context.Images.FindAsync(image.ImageId);
            if (item == null)
            {
                return false;
            }
            item.FileName = image.FileName;
            item.AltText = image.AltText;
            item.FilePath = image.FilePath;

            context.Images.Update(item);
            await context.SaveChangesAsync();
            return true;
        }

        // Responsible for deleating existing table entries
        public async Task<bool> Delete(Image image)
        {
            var item = await context.Images.FindAsync(image.ImageId);
            if (item == null)
            {
                return false;
            }
            context.Images.Remove(item);
            await context.SaveChangesAsync();
            return true;
        }

        // Returns a list of all static content items
        public async Task<List<Image>> List()
        {
            var ImageItems = await context.Images.ToListAsync();
            return ImageItems;
        }

        // Adds a new image to the database and stores the sent image into the assets folder
        public async Task<bool> Add(ImageViewModel content)
        {
            IFormFile file = content.ImgFile;
            if (file == null)
            {
                return false;
            }

            var fileName = Path.GetFileName(file.FileName); 
            var filePath = Path.Combine("wwwroot/assets", fileName);

            // This part is responsible for storing the input image
            using (var fileStream = new FileStream(filePath, FileMode.Create))
            {
                await file.CopyToAsync(fileStream);
            }
            var image = new Image { FileName = fileName, FilePath = $"/assets/{fileName}", AltText = content.AltText , StaticContentID = content.StaticContentID, DateUploaded = DateTime.Now };

            await this.context.Images.AddAsync(image);
            await this.context.SaveChangesAsync();
            return true;
        }

    }
}
