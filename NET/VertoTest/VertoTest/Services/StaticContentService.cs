using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure.Internal;
using System.Diagnostics;
using VertoTest.Context;
using VertoTest.Models;
using static System.Net.Mime.MediaTypeNames;

namespace VertoTest.Services
{
    public class StaticContentService: StaticContentInterface
    {
        private readonly VertoAppDbContext context;

        public StaticContentService(VertoAppDbContext context)
        {
            this.context = context;
        }

        public async Task<int?> GetNextContentId()
        {
            return await this.context.StaticContents.OrderByDescending(sc => sc.ContentId).Select(sc => sc.ContentId).FirstOrDefaultAsync();
        }

        // Adds a new element to the database
        public async Task<int> Add(StaticContent content)
        {
            // Check to see if foreign key is legal
            Debug.WriteLine(" Exists(content.ContentId) " + await Exists(content.ContentId));
            if (!await ValidateForeignKeys(content, null) || await Exists(content.ContentId))
            {
                return -1;
            }
            Debug.WriteLine("here" + content.Body);
            Debug.WriteLine("here" + content.SliderItemID);
            
            Debug.WriteLine("here" + content.DateUpdated);
            Debug.WriteLine("here" + content.Title);


            content.DateUpdated = DateTime.Now;
            await this.context.StaticContents.AddAsync(content);
            await this.context.SaveChangesAsync();
            return content.ContentId;
        }

        // Returns a sorted list of all static content items
        public async Task<List<StaticContent>> List()
        {
            var staticElements = await this.context.StaticContents.OrderBy(sc => sc.Order).ToListAsync();
            return staticElements;
        }

        // Returns a list sorted of all static content items that match the class name
        public async Task<List<StaticContent>> ListWithClass(String className)
        {
            var staticElements = await this.context.StaticContents.Where(sc => sc.StaticClass == className).OrderBy(sc => sc.Order).ToListAsync();
            return staticElements;
        }

        // Handles table edit/updates  
        public async Task<bool> Edit(StaticContent content)
        {
            Debug.WriteLine("ContentId : " + content.ContentId);
            var item = await context.StaticContents.FindAsync(content.ContentId);
            Debug.WriteLine("item ContentId : " + item.ContentId);
            if (item == null)
            {
                return false;
            }
            // Check to see if foreign key is legal
            Debug.WriteLine("item ContentId : " + item.ContentId);
            if (! await ValidateForeignKeys(content, null))
            {
                return false;
            }
            Debug.WriteLine("item ContentId next : " + item.ContentId);
            item.Title = content.Title;
            item.SliderItemID = content.SliderItemID;
            item.DateUpdated = DateTime.Now;
            item.Body = content.Body;
            item.StaticClass = content.StaticClass;
            item.Order = content.Order;


            await this.context.SaveChangesAsync();
            return true;
        }

        // Checks to see if item already exists in table
        public async Task<bool> Exists(int id)
        {
            return await this.context.StaticContents.AnyAsync(staticContent => staticContent.ContentId == id);
        }

        // Checks that the new data does not violate any database foreign key rules
        private async Task<bool> ValidateForeignKeys(StaticContent content,int? ImageId)
        {
            var imageExists = await this.context.Images.AnyAsync(image => image.ImageId == ImageId);
            Debug.WriteLine("imageExists: " + imageExists);
            if (!imageExists && ImageId != null)
            {
                return false; ;
            }
            var sliderItemExists = await this.context.ContentPanels.AnyAsync(si => si.ContentItemID == content.SliderItemID);
            Debug.WriteLine("sliderItemExists: " + sliderItemExists);
            Debug.WriteLine(content.SliderItemID == null);
            Debug.WriteLine(content.SliderItemID != null);
            Debug.WriteLine(content.SliderItemID != 0);
            if (!sliderItemExists && (content.SliderItemID != null && content.SliderItemID != 0))
            {
                return false;
            }
            Debug.WriteLine("test passed");
            return true;
        }

        // Responsible for deleating existing table entries
        public async Task<bool> Delete(StaticContent content)
        {
            var item = await context.StaticContents.FindAsync(content.ContentId);
            if (item == null)
            {
                return false;
            }
            context.StaticContents.Remove(item);
            await context.SaveChangesAsync();
            return true;
        }

        // Returns the item with a matching primary key
        public async Task<StaticContent?> GetViaId(int id)
        {
            return await this.context.StaticContents.FirstOrDefaultAsync(staticContent => staticContent.ContentId == id);
        }

    }
}
