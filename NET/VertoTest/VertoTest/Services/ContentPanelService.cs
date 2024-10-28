using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Diagnostics;
using VertoTest.Context;
using VertoTest.Models;

namespace VertoTest.Services
{
    public class ContentPanelService : ContentPanelInterface
    {
        private readonly VertoAppDbContext context;
        public ContentPanelService(VertoAppDbContext context)
        {
            this.context = context;
        }

        // Responsible for deleating existing table entries
        public async Task<bool> Delete(ContentPanels content)
        {
            var item = await context.ContentPanels.FindAsync(content.ContentItemID);
            if (item == null)
            {
                return false;
            }
            this.context.ContentPanels.Remove(item);
            await this.context.SaveChangesAsync();
            return true;
        }
        // Handles table edit/updates 
        public async Task<bool> Edit(ContentPanels content)
        {
            var item = await context.ContentPanels.FindAsync(content.ContentItemID);
            if (item == null && content.ContentItemID != null)
            {
                return false;
            }
            item.IsActive = content.IsActive;
            item.Title = content.Title;

            await this.context.SaveChangesAsync();
            return true;
        }
        // Returns a sorted list of all content panels items
        public async Task<List<ContentPanels>> List()
        {
            var sliderItems = await context.ContentPanels.Include(si => si.StaticContents).ThenInclude(sc => sc.Images).OrderBy(sc => sc.Order).ToListAsync();

            // This ensures that the related content panels items are also in order
            foreach (var sliderItem in sliderItems)
            {
                sliderItem.StaticContents = sliderItem.StaticContents.OrderBy(sc => sc.Order).ToList();
            }

            return sliderItems;
        }

        // Adds a new element to the database
        public async Task Add(ContentPanels content)
        {
            await this.context.ContentPanels.AddAsync(content);
            await this.context.SaveChangesAsync();
        }
        // Returns the item with a matching primary key
        public async Task<ContentPanels?> GetViaId(int id)
        {
            return await this.context.ContentPanels.FirstOrDefaultAsync(sliderItem => sliderItem.ContentItemID == id);
        }

        // Checks to see if item already exists in table
        public async Task<bool> Exists(int sliderItemid)
        {
            return await this.context.ContentPanels.AnyAsync(sliderItem => sliderItem.ContentItemID == sliderItemid);
        }
    }
}
