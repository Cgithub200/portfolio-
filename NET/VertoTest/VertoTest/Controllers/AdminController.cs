using Microsoft.AspNetCore.Mvc;
using VertoTest.Context;
using Microsoft.EntityFrameworkCore;
using VertoTest.Models;
using System.Diagnostics;
using VertoTest.Services;


namespace VertoTest.Controllers
{
    public class AdminController : Controller
    {
        private readonly ContentPanelInterface sliderItemInterface;
        private readonly ImageServiceInterface imageServiceInterface;
        private readonly StaticContentInterface staticContentInterface;

        public AdminController(VertoAppDbContext context, ImageServiceInterface imageServiceInterface , StaticContentInterface staticContentInterface, ContentPanelInterface sliderItemInterface)
        {
            this.imageServiceInterface = imageServiceInterface; 
            this.staticContentInterface = staticContentInterface;
            this.sliderItemInterface = sliderItemInterface;
        }

        // Displays the view to create a panel item
        [HttpGet]
        public IActionResult AddStaticElement()
        {
            return View();
        }

        // Allows for the addition of panel components
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> AddStaticElement(StaticContent content)
        {
            Debug.WriteLine("This is a debug message.");
            Debug.WriteLine("This is a debug message." + content.Title);
            await staticContentInterface.Add(content);
            return View();


        }

        // Opens a view to list all components regardless of its panels
        [HttpGet]
        public async Task<IActionResult> StaticElementsList()
        {
            var staticElements = await staticContentInterface.List();
            return View(staticElements);
        }

        // Opens a view for editing panel components
        [HttpGet]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ModStaticElement(int id)
        {

            var element = await staticContentInterface.GetViaId(id);
            if (element == null)
            {
                return NotFound();
            }

            return View(element); 
        }

        // Allows for editing panel components
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ModStaticElement(StaticContent content)
        {
            Debug.WriteLine("Title: " + content.Title);
            Debug.WriteLine("ContentId: " + content.ContentId);
            Debug.WriteLine("Body: " + content.Body);
            Debug.WriteLine("Order: " + content.Order);
            if (!await staticContentInterface.Edit(content))
            {
                return NotFound();
            }

            return RedirectToAction("StaticElementsList", "Admin");
        }

        // Allows for the deleation of panel components
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteStaticElement(StaticContent content)
        {

            Debug.WriteLine("Title: " + content.Title);
            Debug.WriteLine("ContentId: " + content.ContentId);
            Debug.WriteLine("Body: " + content.Body);
            Debug.WriteLine("Order: " + content.Order);



            content = await staticContentInterface.GetViaId(content.ContentId);
            if (content == null)
            {
                return NotFound();
            }

            var success = await staticContentInterface.Delete(content);
            if (!success)
            {
                return NotFound();
            }

            return RedirectToAction("StaticElementsList", "Admin");
        }
            



        // Opens a view to add panels
        [HttpGet]
        public IActionResult AddPanel()
        {
            return View();
        }

        // Allows for new panels to be created
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> AddPanel(ContentPanels content)
        {
            var sliderItem = await sliderItemInterface.Exists(content.ContentItemID);
            if (sliderItem)
            {
                return Conflict("sliderItem with the same ID already exists.");
            }
            await sliderItemInterface.Add(content);
            return View();
        }

        // Returns a list of all panels
        [HttpGet]
        public async Task<IActionResult> PanelList()
        {
            var sliderItems = await sliderItemInterface.List();
            return View(sliderItems);
        }

        // Handles panel edits
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> EditPanel(ContentPanels content)
        {
            
            if (!ModelState.IsValid) { return View("PanelList"); }

            var Success = await sliderItemInterface.Edit(content);
            
            if (!Success)
            {
                return NotFound();
            }
            return RedirectToAction("PanelList");
            
            
        }

        // Handles panel removal
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeletePanel(ContentPanels content)
        {
            content = await sliderItemInterface.GetViaId(content.ContentItemID);
            if (content == null)
            {
                return NotFound();
            }

            var Success = await sliderItemInterface.Delete(content);
            return RedirectToAction("PanelList");
        }


        // View for adding images
        [HttpGet]
        public IActionResult AddImage()
        {
            return View();
        }

        // Handles adding images to database
        [HttpPost]
        [ValidateAntiForgeryToken]

        public async Task<IActionResult> AddImage(ImageViewModel content)
        {

            if (content.ImgFile == null || content.ImgFile.Length == 0)
            {
                return Conflict("Image does not exist");
            }

            await imageServiceInterface.Add(content);
            return View();
        }

        // Returns a list of all images in database
        [HttpGet]
        public async Task<IActionResult> ImageList()
        {
            var ImageItems = await imageServiceInterface.List();
            return View(ImageItems);
        }

        // View for changing existing image database info
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> EditImage(Image content)
        {
            var updateSuccess = await imageServiceInterface.Edit(content);
            if (!updateSuccess)
            {
                return NotFound();
            }
            return RedirectToAction("ImageList", "Admin");
        }

        // Handles the removal of unwanted images from database
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteImage(Image content)
        {
            var updateSuccess = await imageServiceInterface.Delete(content);
            if (!updateSuccess)
            {
                return NotFound();
            }
            return RedirectToAction("ImageList", "Admin");
        }

    }
}
